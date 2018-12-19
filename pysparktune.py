#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:21:52 2018

@author: erdembocugoz
"""

from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext, SparkConf
import os
os.system('rm -rf metastore_db')
os.system('rm -rf __pycache__')


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
als = ALS(userCol="User", itemCol="Movie", ratingCol="rating", nonnegative = True, implicitPrefs = False)

param_grid = ParamGridBuilder() \
           .addGrid(als.rank, [5, 10, 50, 100]) \
           .addGrid(als.maxIter, [5, 50, 100, 200]) \
           .addGrid(als.regParam, [.01, .05, .1, .15]) \
           .build()
           
           
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator,numFolds=3)



data = spark.read.format("csv").option("header", "true")\
    .load("actual_train.csv")
    
from pyspark.sql.types import IntegerType
data = data.withColumn("User", data["User"].cast(IntegerType()))
data = data.withColumn("Movie", data["Movie"].cast(IntegerType()))
data = data.withColumn("rating", data["rating"].cast(IntegerType()))
data =data.drop('_c0')



