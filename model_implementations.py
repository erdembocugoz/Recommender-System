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





def surprise_SVD(train_file,test_file):
    print("SVD")
    fold = [(train_file, test_file)]
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_folds(fold, reader=reader)
    pkf = PredefinedKFold()
    # Algorithm
    algo = SVDpp(n_epochs=30,lr_all=0.001,reg_all=0.001)
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
def pyspark_ALS(df,df_predict,sc):
    print("ALS")
    pred_als = predictions_ALS(df, df_predict, spark_context=sc, rank=8, lambda_=0.081, iterations=24)
    pred_als_arr = pred_als["Rating"].values
    return np.clip(pred_als_arr,1,5)

def model_pyfm(train_actual,predict):
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
