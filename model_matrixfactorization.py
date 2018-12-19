"""
This file contains the models used for movie recommender system created by using information from Exercise 10.

    Models: Matrix factorization with ALS and SGD

Data sets transformed into respected datasets needed for the functions.


"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from itertools import groupby
from sklearn.feature_extraction import DictVectorizer
from implementations import *
from matrix_fact_helpers import *


def implementation_SGD(train, submission_set):
    # Hyperparameters
    gamma = 0.008
    num_features = 20   # K in the lecture notes
    lambda_user = 0.03
    lambda_item = 0.25
    num_epochs = 20     # number of full passes through the train set
    item_features, user_features = matrix_factorization_SGD(train, gamma, num_features, lambda_user, lambda_item, num_epochs)
    pred = predict(item_features, user_features, submission_set)
    return pred

def implementation_ALS(train,submission_set):
    # Hyperparameters
    num_features = 20   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    stop_criterion = 1e-4
    item_features, user_features = matrix_factorization_ALS(train, num_features, lambda_user, lambda_item, stop_criterion)
    pred = predict(item_features, user_features, submission_set)
    return pred
