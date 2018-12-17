from surprise import *
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV

import pandas as pd
import numpy as np

import pickle

def load_data(train_file, submission_file):
    train_file, submission_file, df, df_toBeSubmitted = modify_data(train_file, submission_file)
    reader = Reader(line_format='user item rating', sep=',')
    
    fold = [(train_file, submission_file)]
    trainAndTest = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()

    # Go through 1 fold
    for trainset, testset in  pkf.split(trainAndTest):
        data = trainset
        test = testset
    
    return data, test, df, df_toBeSubmitted

def modify_data(train_file, submission_file):
    df = pd.read_csv(train_file)
    df_test = pd.read_csv(submission_file)
    
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    
    df_test['User'] = df_test['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df_test['Movie'] = df_test['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df_test['Rating'] = df_test['Prediction']
    df_test = df_test.drop(['Id', 'Prediction'], axis=1)
    
    train_file = 'tmp_train.csv'
    header = ['user_id','item_id','rating']
    df.to_csv(train_file, index=False, header=False)
    ratings_df = pd.read_csv('tmp_train.csv',
                              sep=',',
                             names=header,

                             dtype={
                               'user_id': np.int32,
                               'item_id': np.int32,
                               'rating': np.float32,
                               'timestamp': np.int32,
                             })

    test_file = 'tmp_test.csv'
    header = ['user_id','item_id','rating']
    df_test.to_csv(test_file, index=False, header=False)
    ratings_df_test = pd.read_csv('tmp_test.csv',
                              sep=',',
                             names=header,

                             dtype={
                               'user_id': np.int32,
                               'item_id': np.int32,
                               'rating': np.float32,
                               'timestamp': np.int32,
                             })
    
    return 'tmp_train.csv', 'tmp_test.csv', df, df_test  
            
##################################################################################### 
            
def tuneHyperParams(algtype, trainset, testset, df, param_grid):
    #TUNE HYPERPARAM VIA GRIDSEARCH
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['User', 'Movie', 'Rating']], reader)
    #trainset, testset = train_test_split(data, test_size=.25, random_state=20)
    gs = GridSearchCV(algtype, param_grid, measures=['rmse'], cv=3)

    model = gs.fit(data)

    # best RMSE score
    #print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    #print(gs.best_params['rmse'])
    
    return gs
            
def fitAndPredict(gs, trainset, testset):
    #FIT AND PREDICT
    #algo = SVD(n_factors=factor ,n_epochs=epoch, lr_all=lr_rate, reg_all=reg_rate)
    algo = gs.best_estimator['rmse']

    model = algo.fit(trainset)


    # Predict
    predictions = algo.test(testset)
    return predictions
            
######################################################################################

def submission_table(original_df, col_userID, col_movie, col_rate):
    """ return table according with Kaggle convention """

    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]

def create_submission_file(submission_filename, predictions, toBeSubmitted):
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        if val > 5:
            pred[i] = 5
        elif val < 1:
            pred[i] = 1
        else:
            pred[i] = val
            
    toBeSubmitted.Rating = (np.round(pred)).astype(int)

    submission = submission_table(toBeSubmitted, 'User', 'Movie', 'Rating')

    file_name = submission_filename + '.csv'
    submission.to_csv(file_name, index=False)
            
######################################################################################## 
            
def runSVD(trainset, testset, toBeSubmitted):
    
    algo = SVD(n_epochs=30,lr_all=0.001,reg_all=0.001)
    
    # Fit
    model = algo.fit(trainset)
    # Predict
    predictions = algo.test(testset)
    # Create Submission file
    create_submission_file('SVD', predictions, toBeSubmitted)
    
    return toBeSubmitted
    
def runSVDpp(trainset, testset, toBeSubmitted):
    
    algo = SVDpp(n_epochs=30,lr_all=0.001,reg_all=0.001)
    
    # Fit
    model = algo.fit(trainset)
    # Predict
    predictions = algo.test(testset)
    # Create Submission file
    create_submission_file('SVDpp', predictions, toBeSubmitted)
    
    return toBeSubmitted
    
    
def runKNN(trainset, testset, toBeSubmitted):

    bsl_options = {'method': 'als','n_epochs': 20}
    sim_options = {'name': 'pearson_baseline'}
    algo = KNNBasic(k=300, bsl_options=bsl_options, sim_options=sim_options)
    
    # Fit
    model = algo.fit(trainset)
    # Predict
    predictions = algo.test(testset)
    # Create Submission file
    create_submission_file('KNN', predictions, toBeSubmitted)
    
    return toBeSubmitted
    
def runSlopeOne(trainset, testset, toBeSubmitted):
    
    algo = SlopeOne()
    
    # Fit
    model = algo.fit(trainset)
    # Predict
    predictions = algo.test(testset)
    # Create Submission file
    create_submission_file('SlopeOne', predictions, toBeSubmitted)
    
    return toBeSubmitted
    
def runBaselineOnly(trainset, testset, toBeSubmitted):
    
    algo = BaselineOnly()
    
    # Fit
    model = algo.fit(trainset)
    # Predict
    predictions = algo.test(testset)  
    # Create Submission file
    create_submission_file('SlopeOne', predictions, toBeSubmitted)
    
    return toBeSubmitted
            
            
            
            

