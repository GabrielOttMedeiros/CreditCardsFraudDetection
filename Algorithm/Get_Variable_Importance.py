import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


def Decision_Tree_Importance(X, Y):
    
    ####################################
    ## Decision Tree Hyper-Parameters ##
    ####################################

    ## Creating an array of max_depths
    max_depth = np.arange(3,11,1)

    ## Creating an array of thresholds
    threshold_list = np.arange(0.05,0.55,0.05)

    ## Defining empty lists to hold the results
    DecisionTreesParametersDepth = []
    DecisionTreesParametersThreshold = []

    ## Here we loop through each list to have every possible combination of hyper-parameters 
    for depth in range(0, len(max_depth)):
        for threshold in range(0, len(threshold_list)):
            DecisionTreesParametersDepth.append(max_depth[depth])
            DecisionTreesParametersThreshold.append(threshold_list[threshold])

    ## Here we create a data frame to hold the results so we can use it for predicitons later
    DecisionTreesParameters = pd.DataFrame({'max_depth':DecisionTreesParametersDepth,
                                          'threshold':DecisionTreesParametersThreshold})
    
    ## Extracting the variable importances in next function
    DTC_results = Extract_Decision_Tree_Importance(X, Y, DecisionTreesParameters)
    return DTC_results


def Extract_Decision_Tree_Importance(X, Y, data):
    
    ################################################
    ## DecisionTreeClassifier Variable Importance ##
    ################################################

    ## Defining empty data frame for results
    results = pd.DataFrame(columns = ['Depth', 'Threshold', 'Acc.', 'Rec.', 'Importances'])
    results['Importances'] = results['Importances'].astype(object)

    n = data.shape[0]

    for i in range(0, n):

        ## Defining empty lists to store model performance results
        accuracy_scores = []
        recall_scores = []
        importances = pd.DataFrame(columns = X.columns)

        for j in range(0, 10):

            ## Splitting the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

            ## Building the model
            md1 = DecisionTreeClassifier(
                max_depth = data.loc[i, 'max_depth']).fit(X_train ,Y_train)

            ## Predicting on the test set
            preds = md1.predict_proba(X_test)[:,1]

            ## Applying the cut-off value
            preds = np.where(preds < data.loc[i, 'threshold'], 0, 1)

            ## Computing the accuracy and recall of the model
            accuracy_scores.append(accuracy_score(Y_test, preds))
            recall_scores.append(recall_score(Y_test, preds))

            importances.loc[j] = md1.feature_importances_.T

        ## Adding the model depth and threshold values to the results data frame
        results.loc[i, 'Depth'] = data.loc[i, 'max_depth']
        results.loc[i, 'Threshold'] = data.loc[i, 'threshold']

        ## Reporting the accuracy and recall of the model
        results.loc[i, 'Acc.'] = np.mean(accuracy_scores)
        results.loc[i, 'Rec.'] = np.mean(recall_scores)

        ## Adding the variable importance array to the results data frame
        coefficients = pd.DataFrame(importances.mean()).T
        results.loc[i, 'Importances'] = coefficients.to_numpy()
    
    ## Creating the performance variable: the average of accuracy and recall
    results['Performance'] = (results['Acc.'] + results['Rec.']) / 2
    
    ## Storing the model with the best performance
    DTC_results = results.sort_values('Performance', ascending = False).loc[0]

    ## Returning the results
    return DTC_results


def Random_Forest_Importance(X, Y):
    
    ####################################
    ## Random Forest Hyper-Parameters ##
    ####################################

    ## Creating an array of max_depths
    max_depth = np.arange(3,11,1)

    ## Creating an array of n_estimators
    n_estimators = np.arange(100,1100,100)

    ## Creating an array of tresholds
    threshold_list = np.arange(0.05,0.55,0.05)

    ## Defining empty lists to hold the results
    RandomForestParametersDepth = []
    RandomForestParametersEstimators = []
    RandomForestParametersThreshold = []

    ## Here we loop through each list to have every possible combination of hyper-parameters 
    for depth in range(0, len(max_depth)):
        for estimator in range(0, len(n_estimators)):
            for threshold in range(0, len(threshold_list)):
                RandomForestParametersDepth.append(max_depth[depth])
                RandomForestParametersEstimators.append(n_estimators[estimator])
                RandomForestParametersThreshold.append(threshold_list[threshold])

    ## Here we create a data frame to hold the results so we can use it for predicitons later
    RandomForestParameters = pd.DataFrame({'max_depth':RandomForestParametersDepth,
                                           'n_estimators':RandomForestParametersEstimators,
                                          'threshold':RandomForestParametersThreshold})
    
    ## Extracting the variable importances in next function
    RFC_results = Extract_Decision_Tree_Importance(X, Y, RandomForestParameters)
    return RFC_results


def Extract_Random_Forest_Importance(X, Y, data):
    
    ################################################
    ## RandomForestClassifier Variable Importance ##
    ################################################

    ## Defining empty data frame for results
    results = pd.DataFrame(columns = ['Depth', 'Estimators', 'Threshold', 'Acc.', 'Rec.', 'Importances'])
    results['Importances'] = results['Importances'].astype(object)

    n = data.shape[0]

    for i in range(0, n):

        ## Defining empty lists to store model performance results
        accuracy_scores = []
        recall_scores = []
        importances = pd.DataFrame(columns = X.columns)

        for j in range(0, 10):

            ## Splitting the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

            ## Building the model
            md1 = RandomForestClassifier(
                max_depth = data.loc[i, 'max_depth'], 
                n_estimators = data.loc[i, 'n_estimators']).fit(X_train ,Y_train)

            ## Predicting on the test set
            preds = md1.predict_proba(X_test)[:,1]

            ## Applying the cut-off value
            preds = np.where(preds < data.loc[i, 'threshold'], 0, 1)

            ## Computing the accuracy and recall of the model
            accuracy_scores.append(accuracy_score(Y_test, preds))
            recall_scores.append(recall_score(Y_test, preds))

            ## Extracting the feature importances
            importances.loc[j] = md1.feature_importances_.T

        ## Adding the model depth, estimators, treshold values to the results data frame
        results.loc[i, 'Depth'] = data.loc[i, 'max_depth']
        results.loc[i, 'Estimators'] = data.loc[i, 'n_estimators']
        results.loc[i, 'Threshold'] = data.loc[i, 'threshold']

        ## Reporting the accuracy and recall of the model
        results.loc[i, 'Acc.'] = np.mean(accuracy_scores)
        results.loc[i, 'Rec.'] = np.mean(recall_scores)

        ## Adding the variable importance array to the results data frame
        coefficients = pd.DataFrame(importances.mean()).T
        results.loc[i, 'Importances'] = coefficients.to_numpy()
        
    ## Creating the performance variable: the average of accuracy and recall
    results['Performance'] = (results['Acc.'] + results['Rec.']) / 2
    
    ## Storing the model with the best performance
    RFC_results = results.sort_values('Performance', ascending = False).loc[0]

    ## Returning the results
    return RFC_results


def Getting_Best_Model(DTC_results, RFC_results, X, Y):
    
    ## Keeping the n most important variables
    n = 10
    
    ## Extracting the performance variable from each data frame
    DTC_Performance = DTC_results['Performance']
    RFC_Performance = RFC_results['Performance']
    
    ## If statement to extract best model
    if (DTC_Performance > RFC_Performance):
        importances = DTC_results['Importances']

    else:
        importances = RFC_results['Importances']
        
    ## Extracting the ten most importance variables
    variables = pd.DataFrame(importances, columns = X.columns)
    
    ## Sorting the columns in order of most important to least
    variables = variables.sort_values(by = 0, axis = 1, ascending = False)
    
    ## Extracting the top-10 variables
    variables = variables.iloc[:, 0:n]
    
    ## Creating the final data set to be returned
    final_data = pd.DataFrame(columns = variables.columns)
    
    ## Keeping the ten most importance variables
    for i in range(0, n):
        final_data.loc[:, final_data.columns[i]] = X.loc[:, final_data.columns[i]]

    ## Adding the Y values to the final data set
    final_data['is_fraud'] = Y
    
    ## Returning the final data set
    return final_data