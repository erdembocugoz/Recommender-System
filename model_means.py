"""
This file contains the models used for movie recommender system created by using information from Exercise 10.

    Models:  user mean, item mean, global mean

Data sets transformed into respected datasets needed for the functions.


"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from itertools import groupby
from sklearn.feature_extraction import DictVectorizer
from implementations import *
from matrix_fact_helpers import *


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
