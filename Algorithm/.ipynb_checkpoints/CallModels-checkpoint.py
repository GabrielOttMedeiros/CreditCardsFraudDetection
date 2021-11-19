import pandas as pd
import numpy as np
import EvaluateModels as em
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


X_train, Y_train = train_data.drop(columns = 'is_fraud'), train_data['is_fraud']


### REMOVE LATER
X_train = X_train.loc[0:10000]
Y_train = Y_train.loc[0:10000]


X = X_train
Y = Y_train



import Get_Variable_Importance as gvi

print('DecisionTrees_Variable_Importance')
decision_trees_variable_importance = gvi.Decision_Tree_Importance(X,Y)

print('RandomForest_Variable_Importance')
random_forest_variable_importance = gvi.Random_Forest_Importance(X,Y)

print('ImportantVariables_Variable_Importance')
df_importance_columns = gvi.Getting_Best_Model(decision_trees_variable_importance,random_forest_variable_importance,X,Y)


X_test = test_data[df_importance_columns.drop(columns = 'is_fraud').columns]
Y_test = test_data['is_fraud']

### REMOVE LATER
X_test = X_test.loc[0:10000]
Y_test = Y_test.loc[0:10000]

X_train = train_data[df_importance_columns.drop(columns = 'is_fraud').columns]
Y_train = train_data['is_fraud']



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
RF_best_model.to_csv('CreditCardsFraudDetection/Algorithm/RF_best_model.csv', index = False)
NN_results.to_csv('CreditCardsFraudDetection/Algorithm/NN_results.csv', index = False)
SVC_results.to_csv('CreditCardsFraudDetection/Algorithm/SVC_results.csv', index = False)
SVC_best_model.to_csv('CreditCardsFraudDetection/Algorithm/SVC_best_model.csv', index = False)
LR_results.to_csv('CreditCardsFraudDetection/Algorithm/LR_results.csv', index = False)
ADA_DTC.to_csv('CreditCardsFraudDetection/Algorithm/ADA_DTC.csv', index = False)
ADA_SVC.to_csv('CreditCardsFraudDetection/Algorithm/ADA_SVC.csv', index = False)
GBC_results.to_csv('CreditCardsFraudDetection/Algorithm/GBC_results.csv', index = False)