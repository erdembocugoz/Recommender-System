"""
This file contains all the models used for movie recommender system.

    Surprise models: baseline, KNN_user, KNN_items, svd, svdpp, slope_one
    PySpark models: ALS
    PyFM models: pylibfm.FM (Matrix Factorization)

All the models all perfectly compatible with the trainset and testset format
provided by the Database module of the Surprise library.

For the some models used from external libraries, data sets transformed into respected datasets

**REMARK** It is not recommended to try pySpark ALS algorith with more than 24 maxIter value because it may give error after 24 iteration.


"""



from surprise import SlopeOne, BaselineOnly, KNNBaseline, SVD, SVDpp, accuracy
import numpy as np
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS
import os
from surprise.model_selection import PredefinedKFold
from surprise import Dataset
from surprise import *
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from implementations import *
from matrix_fact_helpers import *


##### SURPRISE MODELS   ############################################################
########################################################################################################################

def surprise_SVD(train_file,test_file):
    """
    Svd with Surprise library.
    Compute the predictions on a test_set after training on a train_set using the method Svd  from Surprise.
    Args:
        train_file (string): path to created test file
        test_file (string): path to created train file
    Hyperparameters:
        n_factors : The number of factors.
        n_epochs : The number of iteration of the SGD procedure
        lr_all: The learning rate for all
        reg_all : The regularization term for all


    Returns:
        numpy array: predictions
    """
    print("SVD")
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    # Algorithm
    algo = SVD(n_epochs=30,lr_all=0.01,reg_all=0.1)
    for trainset, testset in  pkf.split(data):
        # Train
        algo.fit(trainset)

        # Predict
        predictions = algo.test(testset)
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        pred[i] = val
    return pred
def surprise_SVDpp(train_file,test_file):
    """
    Svd++ with Surprise library.
    Compute the predictions on a test_set after training on a train_set using the method Svd++  from Surprise.
    Args:
        train_file (string): path to created test file
        test_file (string): path to created train file
    Hyperparameters:
        n_factors : The number of factors.
        n_epochs : The number of iteration of the SGD procedure
        lr_'x': The learning rate for 'x'
        reg_'x' : The regularization term for 'x'
    'x':
        bi : The item biases
        bu : The user biases
        qi : The item factors
        yj : The (implicit) item factors
        pu : The user factors


    Returns:
        numpy array: predictions
    """
    print("SVDpp")
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    # Algorithm


    algo = SVDpp(n_epochs=40, n_factors=100, lr_bu=0.01, lr_bi=0.01, lr_pu=0.1, lr_qi=0.1, lr_yj=0.01, reg_bu = 0.05, reg_bi = 0.05, reg_pu = 0.09, reg_qi = 0.1, reg_yj=0.01)
    for trainset, testset in  pkf.split(data):
        # Train
        algo.fit(trainset)

        # Predict
        predictions = algo.test(testset)
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        pred[i] = val
    return pred
def surprise_knn_ib(train_file,test_file):
    """
    Knn itembased with Surprise library.
    Compute the predictions on a test_set after training on a train_set using the method KNNBaseLineOnly from Surprise.
    Args:
        train_file (string): path to created test file
        test_file (string): path to created train file
    Hyperparameters arguemnts:
        k : The (max) number of neighbors to take into account for aggregation
        sim_options (dict) – A dictionary of options for the similarity measure.

    Returns:
        numpy array: predictions
    """
    print("knnIB")
    algo = KNNBaseline(k=300, sim_options={'name': 'pearson_baseline', 'user_based': False})
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    for trainset, testset in  pkf.split(data):
        # Train
        algo.fit(trainset)

        # Predict
        predictions = algo.test(testset)
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        pred[i] = val
    return pred

def surprise_knn_ub(train_file,test_file):
    """
    Knn userbased with Surprise library.
    Compute the predictions on a test_set after training on a train_set using the method KNNBaseLineOnly from Surprise.
    Args:
        train_file (string): path to created test file
        test_file (string): path to created train file
    Hyperparameters :
        k : The (max) number of neighbors to take into account for aggregation
        sim_options (dict) – A dictionary of options for the similarity measure.

    Returns:
        numpy array: predictions
    """
    print("knnUB")
    algo = KNNBaseline(k=300, sim_options={'name': 'pearson_baseline', 'user_based': True})
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    for trainset, testset in  pkf.split(data):
    # Train
        algo.fit(trainset)

    # Predict
        predictions = algo.test(testset)
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        pred[i] = val
    return pred
def surprise_baseline(train_file,test_file):
    """
    Baseline with Surprise library.
    Compute the predictions on a test_set after training on a train_set using the method Baseline from Surprise.
    Args:
        train_file (string): path to created test file
        test_file (string): path to created train file
    Hyperparameters:
        -
    Returns:
        numpy array: predictions
    """
    print("baseline")
    algo = BaselineOnly()
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    for trainset, testset in  pkf.split(data):
    # Train
        algo.fit(trainset)

    # Predict
        predictions = algo.test(testset)
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        pred[i] = val
    return pred
def surprise_slopeOne(train_file,test_file):
    """
    SlopeOne with Surprise library.
    Compute the predictions on a test_set after training on a train_set using the method SlopeOne from Surprise.
    Args:
        train_file (string): path to created test file
        test_file (string): path to created train file
    Hyperparameters:
        -
    Returns:
        numpy array: predictions
    """
    print("slopeone")
    algo = SlopeOne()
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    for trainset, testset in  pkf.split(data):
    # Train
        algo.fit(trainset)

    # Predict
        predictions = algo.test(testset)
    pred = np.zeros(len(predictions))
    for i in range(len(predictions)):
        val = predictions[i].est
        pred[i] = val
    return pred
    ##### PYSPARK ALS   ############################################################
    ########################################################################################################################
    ##### PYSPARK ALS   ############################################################
    ########################################################################################################################
def predictions_ALS(train, test, **kwargs):
    """
    ALS with PySpark.
    Compute the predictions on a test_set after training on a train_set using the method ALS from PySpark.
    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set
        **kwargs: Arbitrary keyword arguments. Passed to ALS.train() (Except for the spark_context)
            spark_context (SparkContext): SparkContext passed from the main program. (Useful when using Jupyter)
            rank (int): Rank of the matrix for the ALS
            lambda (float): Regularization parameter for the ALS
            iterations (int): Number of iterations for the ALS
            nonnegative (bool): Boolean to allow negative values or not.
    Returns:
        pandas.DataFrame: predictions, sorted by (Movie, User)
    """

    # Delete folders that causes troubles
    os.system('rm -rf metastore_db')
    os.system('rm -rf __pycache__')

    # Extract Spark Context from the kwargs
    spark_context = kwargs.pop('spark_context')

    # Convert pd.DataFrame to Spark.rdd
    sqlContext = SQLContext(spark_context)

    train_sql = sqlContext.createDataFrame(train).rdd
    test_sql = sqlContext.createDataFrame(test).rdd

    # Train the model
    model = ALS.train(train_sql, **kwargs)

    # Get the predictions
    data_for_predictions = test_sql.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(data_for_predictions).map(lambda r: ((r[0], r[1]), r[2]))

    # Convert Spark.rdd to pd.DataFrame
    df = predictions.toDF().toPandas()

    # Post processing database
    df['User'] = df['_1'].apply(lambda x: x['_1'])
    df['Movie'] = df['_1'].apply(lambda x: x['_2'])
    df['Rating'] = df['_2']
    df = df.drop(['_1', '_2'], axis=1)
    df = df.sort_values(by=['Movie', 'User'])
    df.index = range(len(df))

    return df
def pyspark_ALS(df,df_predict,sc,r=20,l=0.1,i=24):
    print("ALS")
    pred_als = predictions_ALS(df, df_predict, spark_context=sc, rank=r, lambda_=l, iterations=i)
    pred_als_arr = pred_als["Rating"].values
    return np.clip(pred_als_arr,1,5)

##### PYFM    ############################################################
########################################################################################################################

def model_pyfm(train_actual,predict):
    """
    Matrix Factorization using SGD with pyFM library.
    Compute the predictions on a test_set after training on a train_set using the method Svd++  Matrix Factorization using SGD with pyFM library
    Args:
        train_actual (pandas.DataFrame): train set
        predict (pandas.DataFrame): test set
    Hyperparameters:
        num_factors : The number of factors.
        num_iter : The number of iteration of the SGD procedure
        initial_learning_rate:

    Returns:
        numpy array: predictions
    """
    print("pyfm")
    predict_data,y_predict = create_input_pyfm(predict)
    train_actual_data,y_train_actual = create_input_pyfm(train_actual)
    v = DictVectorizer()
    X_train = v.fit_transform(train_actual_data)
    X_test = v.transform(predict_data)
    # Hyperparameters
    num_factors = 20
    num_iter = 200
    task = 'regression'
    initial_learning_rate = 0.001
    learning_rate_schedule = 'optimal'
    fm = pylibfm.FM(num_factors=num_factors, num_iter=num_iter, task=task,initial_learning_rate=initial_learning_rate,learning_rate_schedule=learning_rate_schedule)
    fm.fit(X_train, y_train_actual)
    preds = fm.predict(X_test)
    return np.clip(preds,1,5)



    ##### Matrix Factorization    ############################################################
    ########################################################################################################################

def implementation_SGD(train, gamma, num_features, lambda_user, lambda_item, num_epochs, submission_set):
    item_features, user_features = matrix_factorization_SGD(train, gamma, num_features, lambda_user, lambda_item, num_epochs)
    pred = predict(item_features, user_features, submission_set)
    return pred

def implementation_ALS(train, num_features, lambda_user, lambda_item, stop_criterion,submission_set):
    item_features, user_features = matrix_factorization_ALS(train, num_features, lambda_user, lambda_item, stop_criterion)
    pred = predict(item_features, user_features, submission_set)
    return pred


##### Baseline Implementations    ############################################################
########################################################################################################################

def implementation_global_mean(train, submission_set):
    """baseline method: use the global mean."""
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    # predict the ratings as global mean

    # copy test set
    pred_set = sp.lil_matrix.copy(submission_set)

    # fill test set with predicted label
    users, items, ratings = sp.find(submission_set)

    for row, col in zip(users, items):
        pred_set[row, col] = global_mean_train
    dummy, dummy, pred = sp.find(pred_set)

    return pred

def implementation_user_mean(train, submission_set):
    """baseline method: use the user means as the prediction."""
    num_users, num_items = train.shape
    user_train_mean = np.zeros((num_users, 1))

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[user_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean[user_index] = nonzeros_train_ratings.mean()
        else:
            continue

    # copy test set
    pred_set = sp.lil_matrix.copy(submission_set)

    # fill test set with predicted label
    users, items, ratings = sp.find(submission_set)

    for row, col in zip(users, items):
        pred_set[row, col] = user_train_mean[row]
    dummy, dummy, pred = sp.find(pred_set)

    return pred


def implementation_item_mean(train, submission_set):
    """baseline method: use item means as the prediction."""
    num_users, num_items = train.shape
    item_train_mean = np.zeros((num_items, 1))

    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[:, item_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean[item_index] = nonzeros_train_ratings.mean()
        else:
            continue

    # copy test set
    pred_set = sp.lil_matrix.copy(submission_set)

    # fill test set with predicted label
    users, items, ratings = sp.find(submission_set)

    for row, col in zip(users, items):
        pred_set[row, col] = item_train_mean[col]
    dummy, dummy, pred = sp.find(pred_set)

    return pred
##########################
