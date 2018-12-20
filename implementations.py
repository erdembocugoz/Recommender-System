"""
In general this file contains functions for general purpose like loading data, converting data frames into
correct forms, and tuning hyper parameters



"""

from surprise import *
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV

import pandas as pd
import numpy as np

import pickle


def submission_table(original_df, col_userID, col_movie, col_rate):
    """
    Converts data frame into submision format
    Args:
        original_df (pandas.Dataframe):
        col_userID (string): column name of User ID column
        col_movie (string): column name of Movie ID column
        col_rate (string): column name of Ratings column
    Returns:
        pandas.Dataframe: df
    """

    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]

def split(df, cut):
	""" split table into train and test set """
	size = df.shape[0]
	keys = list(range(size))
	np.random.shuffle(keys)

	key_cut = int(size * cut)
	train_key = keys[:key_cut]
	test_key = keys[key_cut:]

	test = df.loc[test_key]
	train = df.loc[train_key]

	return train, test

def load_data(fileName):
    """
    Loads data
    Args:
        fileName (string): file path
    Returns:
        pandas.Dataframe: df
    """
    df = pd.read_csv(fileName)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df

def create_input_pyfm(df):
    """
    Creates correct format input for PyFM model
    Args:
        df (pandas.Dataframe): training data frame
    Returns:
        pandas.Dataframe: df
    """
    data = []
    y = list(df.Rating)
    usrs = list(df.User)
    mvies = list(df.Movie)
    for i in range(len(df)):
    	y[i] = float(y[i])
    	data.append({"user_id": str(usrs[i]), "movie_id": str(mvies[i])})
    return data,np.array(y)

def tuneHyperParams(algtype, trainset, testset, df, param_grid):
    """
    Tune Hyper Parameters for Surprise library models
    Args:
        algtype (surprise.prediction_algorithms): type of the surprise algorithm
        trainset(pandas.Dataframe) :
        testset(pandas.Dataframe) :
        df(pandas.Dataframe) :
        param_grid : parameters to try
    Returns:
        surprise.GridSearchCV: gs
    """
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

def load_data_hyper(train_file, submission_file):
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
