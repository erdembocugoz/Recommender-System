import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from surprise import *
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV
from model_implementations import *
from sklearn.linear_model import Ridge


def make_dataframe(filepath):
    df = pd.read_csv(filepath)
    df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    df['Rating'] = df['Prediction']
    df = df.drop(['Id', 'Prediction'], axis=1)
    return df



def make_datasets():
    df_train = make_dataframe("data_train.csv")
    df_predict = make_dataframe("data_test.csv")

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
    train_file = 'tmp_train.csv'
    test_file = 'tmp_test.csv'
    data_train.to_csv(train_file, index=False, header=False)
    data_test.to_csv(test_file, index=False, header=False)

    als_pred = pyspark_ALS(data_train,data_test,spark_context)

    knn_ub_pred = surprise_knn_ub(train_file,test_file)
    knn_ib_pred = surprise_knn_ib(train_file,test_file)
    svd_pred = surprise_SVD(train_file,test_file)
    #pyfm_pred = model_pyfm(data_train,data_test)


    return np.array([knn_ub_pred,knn_ib_pred,svd_pred,als_pred])
def calculate_rmse(real_labels, predictions):
    """Calculate RMSE."""
    return np.linalg.norm(real_labels - predictions) / np.sqrt(len(real_labels))

def get_weights(all_preds,target):
    linreg = Ridge(alpha=0.1, fit_intercept=False)
    linreg.fit(all_preds.T, target)
    weights = dict(zip(["knn_ub_pred","knn_ib_pred","svd_pred","als_pred","pyfm_pred"], linreg.coef_))
    rmse = calculate_rmse(target, linreg.predict(all_preds.T))
    print(rmse)
    return weights,linreg,rmse
def get_als_pred(data_train,data_test,spark_context):
    als_pred = pyspark_ALS(data_train,data_test,spark_context)
    return als_pred
