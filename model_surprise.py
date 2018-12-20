"""
This file contains all the models from Surprise library.

    Surprise models: baseline, KNN_user, KNN_items, svd, svdpp, slope_one


All the models all perfectly compatible with the trainset and testset format
provided by the Database module of the Surprise library.




"""



from surprise import SlopeOne, BaselineOnly, KNNBaseline, SVD, SVDpp, accuracy
import numpy as np
import pandas as pd
from surprise.model_selection import PredefinedKFold
from surprise import Dataset
from surprise import *
from implementations import *

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


    algo = SVDpp(n_epochs=40, n_factors=100,lr_all=0.01, reg_all= 0.01)
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
    algo = KNNBaseline(k=60, sim_options={'name': 'pearson_baseline', 'user_based': False})
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
