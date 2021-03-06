import pandas as pd
import numpy as np
import EvaluateModels as em
from sklearn.model_selection import train_test_split
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


## Reading csv files, index_col = 0 makes the first column of the data to become the index of our pandas data frame
train_data = pd.read_csv(file_content_stream, index_col = 0)
test_data = pd.read_csv(file_content_stream2, index_col = 0)

train_data = train_data.reset_index(drop = True)
test_data = test_data.reset_index(drop = True)

train_data = train_data.select_dtypes(exclude=['object'])
test_data = test_data.select_dtypes(exclude=['object'])

train_data = train_data.dropna()
test_data = test_data.dropna()

## Here we define the target and predictor variables for our Variable Importance extractor
X, Y = train_data.drop(columns = 'is_fraud'), train_data['is_fraud']

## Splitting the data. Creating a test set that is 10% of the data
X_train_sample, X_test_sample, Y_train_sample, Y_test_sample = train_test_split(X, Y, test_size = 0.1, stratify = Y)

## Splitting the data again using the 10% of data (1.5 million rows was too computationally expensive)
## (Plus, still using the stratify function to maintain the correct proportions of fraud)
X_train_final, X_test_final, Y_train_final, Y_test_final = train_test_split(X_test_sample,
                                                                            Y_test_sample, 
                                                                            test_size = 0.2, 
                                                                            stratify = Y_test_sample)


# ## Here we call our variable importance model and its results 
import Get_Variable_Importance as gvi

## Extracting only the most important variables
# df_importance_columns = gvi.Importance(X_test_sample,Y_test_sample)
# df_importance_columns.to_csv('FinalData.csv', index = False)

df_importance_columns = pd.read_csv('FinalData.csv')

## Keeping only the top 10 (or potentially less) variables in the final training and testing sets
X_test = X_test_final[df_importance_columns.drop(columns = 'is_fraud').columns]
Y_test = Y_test_final

X_train = X_train_final[df_importance_columns.drop(columns = 'is_fraud').columns]
Y_train = Y_train_final


## Decision Trees
DTC_results = em.DecisionTreesResults(X_test, X_train, Y_test, Y_train)
DTC_best_model = em.DecisionTreesBestModel(DTC_results)
DTC_results.to_csv('DTC_results.csv', index = False)
DTC_best_model.to_csv('DTC_best_model.csv', index = False)

## Random Forest
RF_results = em.RandomForestResults(X_test, X_train, Y_test, Y_train)
RF_best_model = em.DecisionTreesBestModel(RF_results)
RF_results.to_csv('RF_results.csv', index = False)
RF_best_model.to_csv('RF_best_model.csv', index = False)

## Neural Networks
NN_results = em. NeuralNetworksResults(X_test, X_train, Y_test, Y_train)
NN_results.to_csv('NN_results.csv', index = False)

## SVC
SVC_results = em.SupportVectorMachineResults(X_test, X_train, Y_test, Y_train)
SVC_best_model = em.SvcBestModel(SVC_results)
SVC_results.to_csv('SVC_results.csv', index = False)
SVC_best_model.to_csv('SVC_best_model.csv', index = False)

## Logistic Regression
LR_results = em.LogisticRegressionResults(X_test, X_train, Y_test, Y_train)
LR_results.to_csv('LR_results.csv', index = False)

## AdaBoost Decision Trees
DTC_best_model = pd.read_csv('DTC_best_model.csv')
ADA_DTC = em.AdaBoostDecisionTreesResults(X_test, X_train, Y_test, Y_train, DTC_best_model)
ADA_DTC.to_csv('ADA_DTC.csv', index = False)

## AdaBoost SVC
SVC_best_model = pd.read_csv('SVC_best_model.csv')
ADA_SVC= em.AdaBoostSvmResults(X_test, X_train, Y_test, Y_train, SVC_best_model)
ADA_SVC.to_csv('ADA_SVC.csv', index = False)

## GradientBoosting
GBC_results = em.GradientBoostingResults(X_test, X_train, Y_test, Y_train)
GBC_results.to_csv('GBC_results.csv', index = False)
