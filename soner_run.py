import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import Ridge
from glob import glob
from surprise import Dataset, Reader
import random
from blend import *
%reload_ext autoreload
%autoreload 2

data_train,data_test,data_actual_train,data_actual_predict = make_datasets()

train_file = 'tmp_train.csv'
test_file = 'tmp_test.csv'
data_actual_train.to_csv(train_file, index=False, header=True)
data_actual_predict.to_csv(test_file, index=False, header=True)

from model_matrixfactorization import *
from model_means import *

ratings_train_df = submission_table(data_actual_train, 'User', 'Movie', 'Rating')
ratings_test_df = submission_table(data_actual_predict, 'User', 'Movie', 'Rating')
ratings_train_file = 'ratings_train.csv'
ratings_test_file = 'ratings_test.csv'
ratings_train_df.to_csv(ratings_train_file, index=False, header=True)
ratings_test_df.to_csv(ratings_test_file, index=False, header=True)
ratings_train = load_data_as_scipy(ratings_train_file)
ratings_test = load_data_as_scipy(ratings_test_file)

pred_globalmean = implementation_global_mean(ratings_train, ratings_test)
pred_usermean = implementation_user_mean(ratings_train, ratings_test)
pred_itemmean = implementation_item_mean(ratings_train, ratings_test)

pred_sgd = implementation_SGD(ratings_train ,ratings_test)
pred_als = implementation_ALS(ratings_train,ratings_test)

svdpp_pred = surprise_SVDpp(train_file,test_file)
baseline_pred = surprise_baseline(train_file,test_file)
slopeone_pred = surprise_slopeOne(train_file,test_file)


knn_ub_pred = surprise_knn_ub(train_file,test_file)
knn_ib_pred = surprise_knn_ib(train_file,test_file)
svd_pred = surprise_SVD(train_file,test_file)
