import CallModels as cm

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import HyperParameters as hp
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import boto3

## defining bucket
s3 = boto3.resource('s3')

bucket_name = 'gabriel-de-medeiros'

bucket = s3.Bucket(bucket_name)

## specifying file from the bucket
file_key = 'CleanTrainData.csv'
file_key2 = 'CleanTestData.csv'

bucket_object = bucket.Object(file_key)
bucket_object2 = bucket.Object(file_key2)

file_object = bucket_object.get()
file_object2 = bucket_object2.get()

file_content_stream = file_object.get('Body')
file_content_stream2 = file_object2.get('Body')


## Reading csv file, index_col = 0 makes the first column of the data to become the index of our pandas data frame
train_data = pd.read_csv(file_content_stream, index_col = 0)
test_data = pd.read_csv(file_content_stream2, index_col = 0)

train_data = train_data.reset_index(drop = True)
test_data = test_data.reset_index(drop = True)

train_data = train_data.select_dtypes(exclude=['object'])
test_data = test_data.select_dtypes(exclude=['object'])

train_data = train_data.dropna()
test_data = test_data.dropna()

X_train, X_test,Y_train, Y_test = train_data.iloc[:,0:39], test_data.iloc[:,0:39], train_data['is_fraud'], test_data['is_fraud']

## Decision Trees
DTC_results = cm.DecisionTreesResults(X_test, X_train, Y_test, Y_train)
DTC_best_model = cm.DecisionTreesBestModel(DTC_results)

## Random Forest
RF_results = cm.RandomForestResults(X_test, X_train, Y_test, Y_train)
RF_best_model = cm.DecisionTreesBestModel(RF_results)

## Neural Networks
NN_results = cm. NeuralNetworksResults(X_test, X_train, Y_test, Y_train)

## SVC
SVC_results = cm.SupportVectorMachineResults(X_test, X_train, Y_test, Y_train)
SVC_best_model = cm.SvcBestModel(SVC_results)

## Logistic Regression
LR_results = cm.LogisticRegressionResults(X_test, X_train, Y_test, Y_train)

## AdaBoost Decision Trees
ADA_DTC = cm.AdaBoostDecisionTreesResults(X_test, X_train, Y_test, Y_train, best_model)

## AdaBoost SVC
ADA_SVC= cm.AdaBoostSvmResults(X_test, X_train, Y_test, Y_train, best_model)

## GradientBoosting
GBC_results = cm.GradientBoostingResults(X_test, X_train, Y_test, Y_train)