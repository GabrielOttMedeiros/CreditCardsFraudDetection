import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import HyperParameters as hp

## Creating an array of thresholds
threshold_list = np.arange(0.05,0.55,0.05)
N = len(threshold_list)


def Importance(X, Y):
    
    ####################################
    ## Extracting Variable Importance ##
    ####################################

    ## Getting the proper hyper-parameters
    DecisionTreesParameters = hp.DecisionTreesHyperParameters()
    RandomForestParameters = hp.RandomForestHyperParameters()
    
    ## Extracting the variable importances in next function
    DTC_results = Extract_Decision_Tree_Importance(X, Y, DecisionTreesParameters)
    RFC_results = Extract_Random_Forest_Importance(X, Y, RandomForestParameters)
    
    ## Finding the best model and their importances
    final_data = Getting_Best_Model(DTC_results, RFC_results, X, Y)
    
    ## Returning the final data set
    return final_data


def Extract_Decision_Tree_Importance(X, Y, data):
    
    ################################################
    ## DecisionTreeClassifier Variable Importance ##
    ################################################

    ## Defining empty data frame for results
    list_results = []

    n = data.shape[0]
    
    print('Decision Trees:')
    
    for i in tqdm(range(0, n)):
        
        ## Splitting the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

        ## Building the model
        md1 = DecisionTreeClassifier(max_depth = data.loc[i, 'max_depth']).fit(X_train ,Y_train)
        
        ## Predicting on the test set
        preds = md1.predict_proba(X_test)[:,1]

        for j in range(0, N):

            ## Applying the cut-off value
            preds_new = np.where(preds < threshold_list[j], 0, 1)
            
            ## Appending model information to the list
            list_results.append([data.loc[i, 'max_depth'], 
                                threshold_list[j], 
                                accuracy_score(Y_test, preds_new),
                                recall_score(Y_test, preds_new),
                                md1.feature_importances_.T])
            
    ## Defining a new data frame
    DTC_results = pd.DataFrame(columns = ['Depth', 'Threshold', 'Acc.', 'Rec.', 'Importances'],
                              data = list_results)

    ## Creating the performance variable: the average of accuracy and recall
    DTC_results['Performance'] = 2 / ((1/DTC_results['Acc.']) + (1/DTC_results['Rec.']))
    
    ## Storing the model with the best performance
    DTC_results = DTC_results.sort_values('Performance', ascending = False).loc[0]

    ## Returning the results
    return DTC_results


def Extract_Random_Forest_Importance(X, Y, data):
    
    ################################################
    ## RandomForestClassifier Variable Importance ##
    ################################################

    ## Defining empty data frame for results
    list_results = []

    n = data.shape[0]
    
    print('Random Forest:')

    for i in tqdm(range(0, n)):
        
        ## Splitting the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

        ## Building the model
        md1 = RandomForestClassifier(max_depth = data.loc[i, 'max_depth'], 
            n_estimators = data.loc[i, 'n_estimators']).fit(X_train ,Y_train)

        ## Predicting on the test set
        preds = md1.predict_proba(X_test)[:,1]

        for j in range(0, N):
            
            ## Applying the cut-off value
            preds_new = np.where(preds < threshold_list[j], 0, 1)
            
            ## Appending model information to the list
            list_results.append([data.loc[i, 'max_depth'], 
                                 data.loc[i, 'n_estimators'], 
                                 threshold_list[j], 
                                 accuracy_score(Y_test, preds_new), 
                                 recall_score(Y_test, preds_new), 
                                 md1.feature_importances_.T])
            
    ## Defining a new data frame
    RFC_results = pd.DataFrame(columns = ['Depth', 'Estimators', 'Threshold', 'Acc.', 'Rec.', 'Importances'], 
                               data = list_results)

    ## Creating the performance variable: the average of accuracy and recall
    RFC_results['Performance'] = 2 / ((1/RFC_results['Acc.']) + (1/RFC_results['Rec.']))
    
    ## Storing the model with the best performance
    RFC_results = RFC_results.sort_values('Performance', ascending = False).loc[0]

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
    variables = pd.DataFrame(importances).transpose()
    variables.columns = X.columns
    
    ## Sorting the columns in order of most important to least
    variables = variables.sort_values(by = 0, axis = 1, ascending = False)
    
    ## Extracting the top-10 variables
    variables = variables.iloc[:, 0:n]
    
    ## Removing variables with a zero importance
    for i in range(0, n):
        if (variables.loc[0, variables.columns[i]] != 0):
            new_n = i + 1

    variables = variables.iloc[:, 0:new_n]
    
    ## Creating the final data set to be returned
    final_data = pd.DataFrame(columns = variables.columns)
    
    ## Keeping the ten most importance variables
    for i in range(0, new_n):
        final_data.loc[:, final_data.columns[i]] = X.loc[:, final_data.columns[i]]

    ## Adding the Y values to the final data set
    final_data['is_fraud'] = Y
    
    ## Returning the final data set
    return final_data
