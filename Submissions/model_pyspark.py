"""
This file contains the model used for movie recommender system from PySpark Library.

    PySpark models: ALS

Data sets transformed into respected datasets needed for the library.

**REMARK** It is not recommended to try pySpark ALS algorith with more than 24 maxIter value because it may give error after 24 iteration.


"""




import numpy as np
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS
import os
from sklearn.feature_extraction import DictVectorizer
from implementations import *





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
