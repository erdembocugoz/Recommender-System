"""
This file contains hyper paramater tuning functions for collabrative filtering models.

Hyper parameter tuning for surprise model : it is an general function should work with all Surprise models.
Hyper parameter tuning for pySpark model: uses cross validation to find hyper parameters giving the optimal rmse.


**REMARK** It is not recommended to try pySpark ALS algorith with more than 24 maxIter value because it may give error after 24 iteration.
"""


from surprise import *
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV

from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType



import pandas as pd
import numpy as np
import math
from implementations import *

def tune_surprise_model(train_file,test_file,surprise_model,param_grid):
    """
    Tune hyper parameters by using cross validation
    Args:
        train_file (string): path to train csv file
        test_file (string): path to train csv file
        surprise_model (surprise.prediction_algorithms.X ) : X refers to surprise algorithm
                                                             example: surprise_model = SVD()
        param_grid : hyper paramters to try for tuning
                    example for SVD paramater grid: param_grid = {'n_epochs': [50], 'lr_all': [0.01, 0.0005, 0.001,
                                                        0.01, 0.1],'reg_all':[0.01, 0.0005, 0.001, 0.01, 0.1]}


    Returns:
        dict: best_params
        dict: best_score
        urprise.prediction_algorithms."MODEL NAME": model
    """
    trainset, testset, df, toBeSubmitted = load_data_hyper(train_file, test_file)
    algotype = surprise_model
    gs = tuneHyperParams(algotype, trainset, testset, df, param_grid)
    best_params = gs.best_params
    best_score = gs.best_score
    model = p.best_estimator['rmse']

    return  best_params,best_score,model

def tune_pyspark_als(train_file,test_file,param_grid):
    """
    Tune hyper parameters by using cross validation
    Args:
        train_file (string): path to train csv file
        test_file (string): path to train csv file
        param_grid() : hyper paramters to try for tuning
                    example:param_grid = ParamGridBuilder() \
                           .addGrid(als.rank, [1,5,10,15]) \
                           .addGrid(als.maxIter, [24]) \
                           .addGrid(als.regParam, [.01]) \
                           .build()

    Returns:
        dict: best_params
        dict: best_score
        pyspark.ml.tuning.CrossValidatorModel: model contains best parameters
    """
    spark = SparkSession.builder.appName('Sample').getOrCreate()
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression()
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    als = ALS(userCol="User", itemCol="Movie", ratingCol="rating", nonnegative = True, implicitPrefs = False)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator,numFolds=3)
    data_train,data_test,data_actual_train,data_actual_predict = make_datasets()
    data_actual_train.to_csv("actual_train.csv")
    data = spark.read.format("csv").option("header", "true")\
    .load("actual_train.csv")
    data = data.withColumn("User", data["User"].cast(IntegerType()))
    data = data.withColumn("Movie", data["Movie"].cast(IntegerType()))
    data = data.withColumn("rating", data["rating"].cast(IntegerType()))
    data =data.drop('_c0')
    cvModel = cv.fit(data)
    return cvModel
