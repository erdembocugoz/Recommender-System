"""
This file contains functions for blending models.



**REMARK** It is not recommended to try pySpark ALS algorith with more than 24 maxIter value because it may give error after 24 iteration.
"""





import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from surprise import *
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV
from model_pyfm import *
from model_surprise import *
from model_pyspark import *
from model_means import *
from model_matrixfactorization import *
from matrix_fact_helpers import *

from sklearn.linear_model import Ridge


def make_dataframe(filepath):
    """
    Reads csv file and transform it into 3 column(User,Movie,Rating) data frame
    Args:
        filepath (string): path to csv file
    Returns:
        pandas.Dataframe: df
    """
    df = pd.read_csv(filepath)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df



def make_datasets(train_path="data_train.csv",predict_path="data_test.csv"):
    """
    Reads training csv file and splits it into training and testing set as pandas dataframe, for blending(voting) models
    Reads to be predicted csv file as pandas dataframe
    Converts all dataframes

    Args:
        train_path (string): path to training csv file
        predict_path (string): path to be predicted csv file
    Returns:
        pandas.Dataframe: data_train : train dataframe from train-test split method
        pandas.Dataframe: data_test : test dataframe from train-test split method
        pandas.Dataframe: data_actual_train : actual training data for final predictions
        pandas.Dataframe: data_actual_predict : final data to be predicted
    """
    df_train = make_dataframe(train_path)
    df_predict = make_dataframe(predict_path)

    X_train, X_test, y_train, y_test = train_test_split(df_train[['User','Movie']], df_train['Rating'], test_size=0.5, random_state=56)
    data_train = X_train.join(y_train)
    data_test = X_test.join(y_test)
    data_actual_train = df_train
    data_actual_predict = df_predict
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    data_train.to_csv(train_file, index=False, header=False)
    data_test.to_csv(train_file, index=False, header=False)


    return data_train,data_test,data_actual_train,data_actual_predict

def get_predicts(data_train,data_test,spark_context):
    """
    Train models with training data
    Predict test data with trained models
    Args:
        data_train (pandas.DataFrame): train set
        data_test (pandas.DataFrame): test set for predictions
    Returns:
        numpy.array: all_predictions : predictions obtained from all models
    """
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    data_train.to_csv(train_file, index=False, header=False)
    data_test.to_csv(test_file, index=False, header=False)

    #ratings_train_df = submission_table(data_train, 'User', 'Movie', 'Rating')
    #ratings_test_df = submission_table(data_test, 'User', 'Movie', 'Rating')
    #ratings_train_file = 'ratings_train.csv'
    #ratings_test_file = 'ratings_test.csv'
    #ratings_train_df.to_csv(ratings_train_file, index=False, header=True)
    #ratings_test_df.to_csv(ratings_test_file, index=False, header=True)

    als_pred = pyspark_ALS(data_train,data_test,spark_context)



    #ratings_train = load_data_as_scipy(ratings_train_file)
    #ratings_test = load_data_as_scipy(ratings_test_file)


    #pred_globalmean = implementation_global_mean(ratings_train, ratings_test)
    #pred_usermean = implementation_user_mean(ratings_train, ratings_test)
    #pred_itemmean = implementation_item_mean(ratings_train, ratings_test)

    #pred_sgd = implementation_SGD(ratings_train ,ratings_test)
    #pred_als = implementation_ALS(ratings_train,ratings_test)





    svdpp_pred = surprise_SVDpp(train_file,test_file)
    pyfm_pred = pyfm_predict(data_train,data_test)


    baseline_pred = surprise_baseline(train_file,test_file)
    slopeone_pred = surprise_slopeOne(train_file,test_file)

    knn_ub_pred = surprise_knn_ub(train_file,test_file)
    knn_ib_pred = surprise_knn_ib(train_file,test_file)
    svd_pred = surprise_SVD(train_file,test_file)

    all_predictions = np.array([baseline_pred,slopeone_pred,knn_ub_pred,knn_ib_pred,svd_pred,svdpp_pred,als_pred,pyfm_pred])
    #,pred_globalmean,pred_usermean,pred_itemmean,pred_sgd,pred_als])

    return all_predictions
def calculate_rmse(real_labels, predictions):
    """Calculate RMSE."""
    return np.linalg.norm(real_labels - predictions) / np.sqrt(len(real_labels))

def get_weights(all_preds,target):
    """
    Ensemble models by using voting/Blending
    Establish weights for each model by using Ridge Regression
    Args:
        all_preds (numpy.array): array of prediction arrays obtained from each model
        target (numpy.array): rankings obtained from test score, to calculate rmse
    Returns:
        weights: dict : predictions obtained from all models
        linreg: scikitlearn.Ridge : Ridge Regression model
        rmse: final rmse after blending models
    """
    linreg = Ridge(alpha=0.1, fit_intercept=False)
    linreg.fit(all_preds.T, target)
    weights = dict(zip(["baseline_pred","slopeone_pred","knn_ub_pred","knn_ib_pred","svd_pred","svdpp_pred","als_pred","pyfm_pred","pred_globalmean","pred_usermean","pred_itemmean","pred_sgd","pred_als"], linreg.coef_))
    rmse = calculate_rmse(target, linreg.predict(all_preds.T))
    print(rmse)
    return weights,linreg,rmse

# def get_als_pred(data_train,data_test,spark_context,r,l,i):
#     als_pred = pyspark_ALS(data_train,data_test,spark_context,r,l,i)
#     return als_pred
