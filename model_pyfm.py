"""
This file contains the model used for movie recommender system from PyFM Library.

    PyFM models: pylibfm.FM (Matrix Factorization)

Data sets transformed into respected datasets needed for the library.


"""



import numpy as np
import pandas as pd
import os
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from implementations import *


def pyfm_predict(train_actual,predict):
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
