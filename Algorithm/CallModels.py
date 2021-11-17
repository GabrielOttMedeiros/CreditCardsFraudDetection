from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import EvaluateModels as em
import tensorflow as tf
import pandas as pd
import numpy as np
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
DTC_results = em.DecisionTreesResults(X_test, X_train, Y_test, Y_train)
DTC_best_model = em.DecisionTreesBestModel(DTC_results)

## Random Forest
RF_results = em.RandomForestResults(X_test, X_train, Y_test, Y_train)
RF_best_model = em.DecisionTreesBestModel(RF_results)

## Neural Networks
NN_results = em. NeuralNetworksResults(X_test, X_train, Y_test, Y_train)

## SVC
SVC_results = em.SupportVectorMachineResults(X_test, X_train, Y_test, Y_train)
SVC_best_model = em.SvcBestModel(SVC_results)

## Logistic Regression
LR_results = em.LogisticRegressionResults(X_test, X_train, Y_test, Y_train)

## AdaBoost Decision Trees
ADA_DTC = em.AdaBoostDecisionTreesResults(X_test, X_train, Y_test, Y_train, DTC_best_model)

## AdaBoost SVC
ADA_SVC= em.AdaBoostSvmResults(X_test, X_train, Y_test, Y_train, SVC_best_model)

## GradientBoosting
GBC_results = em.GradientBoostingResults(X_test, X_train, Y_test, Y_train)



DTC_results.to_csv('CreditCardsFraudDetection/Algorithm/DTC_results.csv', index = False)
DTC_best_model.to_csv('CreditCardsFraudDetection/Algorithm/DTC_best_model.csv', index = False)
RF_results.to_csv('CreditCardsFraudDetection/Algorithm/RF_results.csv', index = False)
RF_best_model.to_csv('CreditCardsFraudDetection/Algorithm/RF_best_model.csv'n index = False)
NN_results.to_csv('CreditCardsFraudDetection/Algorithm/NN_results.csv', index = False)
SVC_results.to_csv('CreditCardsFraudDetection/Algorithm/SVC_results.csv', index = False)
SVC_best_model.to_csv('CreditCardsFraudDetection/Algorithm/SVC_best_model.csv', index = False)
LR_results.to_csv('CreditCardsFraudDetection/Algorithm/LR_results.csv', index = False)
ADA_DTC.to_csv('CreditCardsFraudDetection/Algorithm/ADA_DTC.csv', index = False)
ADA_SVC.to_csv('CreditCardsFraudDetection/Algorithm/ADA_SVC.csv', index = False)
GBC_results.to_csv('CreditCardsFraudDetection/Algorithm/GBC_results.csv', index = False)

