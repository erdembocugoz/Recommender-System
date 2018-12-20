import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import Ridge
from glob import glob
from surprise import Dataset, Reader
import random
from blend import *
from implementations import *
from pyspark import SparkContext, SparkConf

print("Loading data sets...")
data_train,data_test,data_actual_train,data_actual_predict = make_datasets()

#Start spark context
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)

print("Getting predictions from each model...")
all_predicts = get_predicts(data_actual_train,data_actual_predict,sc)

print("Loading pre trained Ridge Regression Function for blending models...")
#Load predefined regression function for blending
with open(r"linreg.pkl", "rb") as input_file:
    linreg = pickle.load(input_file)

print("Blending...")
#Blend models
final_predictions = linreg.predict(all_predicts.T)
final_predictions = np.clip(final_predictions, 1, 5)
final_predictions = np.round(final_predictions)

data_actual_predict["Rating"] = final_predictions
print("Creating submission file")
submission = submission_table(data_actual_predict, 'User', 'Movie', 'Rating')


file_name = 'submission.csv'
submission.to_csv(file_name, index=False)

print("Submission file created")
