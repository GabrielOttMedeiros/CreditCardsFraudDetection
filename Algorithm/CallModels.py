import CallModels as cm

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
