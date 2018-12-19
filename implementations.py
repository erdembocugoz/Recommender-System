# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import scipy.sparse as sp
from helpers import *

########################################################################################
#EXPLORATORY DATA ANALYSIS

def calculate_statistics_per_user(ratings):
    user_means = []
    user_stds = []

    num_items, num_users = ratings.shape

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        user_ratings = ratings[:, user_index]
        nonzeros_user_ratings = user_ratings[user_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_user_ratings.shape[0] != 0:
            # calculate mean
            user_ratings_mean = nonzeros_user_ratings.mean()
            user_means.append(user_ratings_mean)
            # calculate std
            dense_arr = np.asarray(nonzeros_user_ratings.todense())
            user_ratings_std = np.sqrt(np.mean((dense_arr - nonzeros_user_ratings.mean())**2))
            user_stds.append(user_ratings_std)
        else:
            continue
    return user_means, user_stds

########################################################################################
#SPLITING DATA INTO TRAIN AND TEST 

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    #valid_ratings = ratings[valid_items, :][: , valid_users]
    valid_ratings = ratings
    
    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format(
        (num_rows, num_cols)))

    nz_items, nz_users = valid_ratings.nonzero()
    
    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test


########################################################################################
#BASELINE

def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    # find the non zero ratings in the test
    nonzero_test = test[test.nonzero()].todense()

    # predict the ratings as global mean
    mse = calculate_mse(nonzero_test, global_mean_train)
    rmse = np.sqrt(1.0 * mse / nonzero_test.shape[1])
    print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))
    
def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    mse = 0
    num_items, num_users = train.shape

    for user_index in range(num_users):
        # find the non-zero ratings for each user in the training dataset
        train_ratings = train[:, user_index]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]
        
        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            user_train_mean = nonzeros_train_ratings.mean()
        else:
            continue
        
        # find the non-zero ratings for each user in the test dataset
        test_ratings = test[:, user_index]
        nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nonzeros_test_ratings, user_train_mean)
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of the baseline using the user mean: {v}.".format(v=rmse))
    
def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    for item_index in range(num_items):
        # find the non-zero ratings for each item in the training dataset
        train_ratings = train[item_index, :]
        nonzeros_train_ratings = train_ratings[train_ratings.nonzero()]

        # calculate the mean if the number of elements is not 0
        if nonzeros_train_ratings.shape[0] != 0:
            item_train_mean = nonzeros_train_ratings.mean()
        else:
            continue
        
        # find the non-zero ratings for each movie in the test dataset
        test_ratings = test[item_index, :]
        nonzeros_test_ratings = test_ratings[test_ratings.nonzero()].todense()
        
        # calculate the test error 
        mse += calculate_mse(nonzeros_test_ratings, item_train_mean)
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of the baseline using the item mean: {v}.".format(v=rmse))
    
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
        

########################################################################################
#MATRIX FACTORIZATION - SGD

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
        
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))

def tune_matrix_factorization_SGD(train, test, gamma, num_features, lambda_user, lambda_item, num_epochs):
    """matrix factorization by SGD."""
    # define parameters
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))

def matrix_factorization_SGD(train, gamma, num_features, lambda_user, lambda_item, num_epochs):
    """matrix factorization by SGD."""
    # define parameters
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))


    #print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        #print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    # evaluate the test error
    #rmse = compute_error(test, user_features, item_features, nz_test)
    #print("RMSE on test data: {}.".format(rmse))
    
    return item_features, user_features

########################################################################################
#MATRIX FACTORIZATION - ALS

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features

def tune_matrix_factorization_ALS(train, test, num_features, lambda_user, lambda_item, stop_criterion):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("\nstart the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    # evaluate the test error
    
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)
    print("test RMSE after running ALS: {v}.".format(v=rmse))
    
def matrix_factorization_ALS(train, num_features, lambda_user, lambda_item, stop_criterion):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    #print("\nstart the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        #print("RMSE on training set: {}.".format(error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])
    
    return item_features, user_features


########################################################################################
#PREDICTION

def predict(item_features, user_features, test_set):
    """ Apply MF model. Multiply matrices W and Z and select the wished
    predictions according to test set
    """

    # copy test set
    pred_set = sp.lil_matrix.copy(test_set)

    # fill test set with predicted label
    users, items, ratings = sp.find(test_set)

    for row, col in zip(users, items):       
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        pred_set[row, col] = user_info.T.dot(item_info)
    dummy, dummy, pred = sp.find(pred_set)
    return pred

def implementation_SGD(train, gamma, num_features, lambda_user, lambda_item, num_epochs, submission_set):
    item_features, user_features = matrix_factorization_SGD(train, gamma, num_features, lambda_user, lambda_item, num_epochs)
    pred = predict(item_features, user_features, submission_set)
    return pred

def implementation_ALS(train, num_features, lambda_user, lambda_item, stop_criterion,submission_set):
    item_features, user_features = matrix_factorization_ALS(train, num_features, lambda_user, lambda_item, stop_criterion)
    pred = predict(item_features, user_features, submission_set)
    return pred
 



































