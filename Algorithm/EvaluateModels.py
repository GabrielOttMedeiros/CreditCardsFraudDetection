## Importing algorithms and libraries
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import HyperParameters as hp
from sklearn.svm import SVC
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np

## Creating cut offs
cut_off_list = np.arange(0.001,0.6,0.001)

## Len of cut off for lists
N = len(cut_off_list)

####################
## Decision Trees ##
####################
def DecisionTreesResults(X_test, X_train, Y_test, Y_train):
    
    ## Empty list to store results
    list_results = []

    ## Here we loop through len of parameters
    DTC_n = hp.DecisionTreesHyperParameters().shape[0]

    print('DecisionTrees')
    for i in tqdm(range(0, DTC_n)):
        
        ## Here we generate our model
        DTC = DecisionTreeClassifier(max_depth = hp.DecisionTreesHyperParameters().loc[i, 'max_depth']).fit(X_train,Y_train)
        
        ## Here we make predictions on our test dataset
        DTC_preds = DTC.predict_proba(X_test)[:,1]
              
        for k in range(0,N):            
            

            
            ## Here we loop though different cut offs
            DTC_preds_new = np.where(DTC_preds < cut_off_list[k],0,1)
            
            ## Here we append the different results
            list_results.append([hp.DecisionTreesHyperParameters().loc[i,'max_depth'],
                                 cut_off_list[k],
                                 accuracy_score(Y_test,DTC_preds_new),
                                 recall_score(Y_test, DTC_preds_new)])

    ## Here we store the results in a data frame
    DTC_results = pd.DataFrame(columns = ['max_depth','cut_off','accuracy','recall'], data = list_results)
        
    ## Here we compute the harmonic average of the model    
    DTC_results['Performance'] = 2/(1/DTC_results['accuracy'] + 1/DTC_results['recall'])   

    ## Here we return the data set of results from this function
    return DTC_results

##############################
## Decision Trees Best Model##
##############################

def DecisionTreesBestModel(DTC_results):
    
    best_model = DTC_results[DTC_results['Performance'] == max(DTC_results['Performance'])]
    
    return best_model


###################
## Random Forest ##
###################
def RandomForestResults(X_test, X_train, Y_test, Y_train):
    
    ## Empty list to store results
    list_results = []

    ## Here we loop through len of parameters
    RF_n = hp.RandomForestHyperParameters().shape[0]

    print('RandomForest')
    for i in tqdm(range(0, RF_n)):
        
        ## Here we generate our model
        RF = RandomForestClassifier(max_depth = hp.RandomForestHyperParameters().loc[i, 'max_depth'],
                                   n_estimators =  hp.RandomForestHyperParameters().loc[i, 'n_estimators']).fit(X_train,Y_train)
        
        ## Here we make predictions on our test dataset
        RF_preds = RF.predict_proba(X_test)[:,1]
        
        
        for k in range(0,N):

            ## Here we loop though different cut offs
            RF_preds_new = np.where(RF_preds < cut_off_list[k],0,1)
            
            ## Here we append the different results
            list_results.append([hp.RandomForestHyperParameters().loc[i, 'max_depth'],
                               hp.RandomForestHyperParameters().loc[i, 'n_estimators'],
                               cut_off_list[k],
                               accuracy_score(Y_test,RF_preds_new),
                               recall_score(Y_test, RF_preds_new)])
    
    ## Here we store the results in a data frame
    RF_results = pd.DataFrame(columns = ['max_depth','n_estimators','cut_off','accuracy','recall'], data = list_results)
     
    ## Here we compute the harmonic average of the model    
    RF_results['Performance'] = 2/(1/RF_results['accuracy'] + 1/RF_results['recall']) 
    
    ## Here we return the data set of results from this function
    return RF_results

#############################
## Random Forest Best Model##
#############################

def DecisionTreesBestModel(RF_results):
    
    best_model = RF_results[RF_results['Performance'] == max(RF_results['Performance'])]
    
    return best_model

#####################
## Neural Networks ##
#####################

def NeuralNetworksResults(X_test, X_train, Y_test, Y_train):
    
    ## Empty list to store results
    list_results = []
    
    NnInputParameters = hp.NeuralNetworksHyperParameters()
    
    ## Here we specify the input dim
    number_of_columns = len(X_train.columns)
    
    ## Here we loop through len of parameters
    NN_n = NnInputParameters.shape[0]

    print('NeuralNetworks')
    for i in tqdm(range(0, NN_n)):
        
        ## Here we generate our model
        NN_md = tf.keras.models.Sequential([     
      tf.keras.layers.Dense(NnInputParameters.loc[i, 'number_of_neurons'], input_dim = number_of_columns, activation = NnInputParameters.loc[i,'activation']),
          
      tf.keras.layers.Dense(2, activation = NnInputParameters.loc[i, 'activation2'])])
        
        NN_md.compile(optimizer = NnInputParameters.loc[i, 'optimizer'], metrics = ['accuracy'], loss=NnInputParameters.loc[i, 'loss_function'])
    
        NN_md.fit(X_train,tf.keras.utils.to_categorical(Y_train, num_classes = 2), epochs = 100,batch_size = 500,verbose = 0,
            validation_data = (X_test,tf.keras.utils.to_categorical(Y_test,num_classes = 2)))
        
        ## Here we make predictions on our test dataset
        NN_preds = NN_md.predict(X_test)[:,1]
        
        for k in range(0,N):
            

    
            ## Here we loop though different cut offs
            NN_preds_new = np.where(NN_preds < cut_off_list[k], 0, 1)
            
            ## Here we append the different results
            list_results.append([NnInputParameters.loc[i, 'number_of_neurons'],
                                NnInputParameters.loc[i, 'activation'],
                                NnInputParameters.loc[i, 'activation2'],
                                cut_off_list[k],
                                accuracy_score(Y_test,NN_preds_new),
                                recall_score(Y_test, NN_preds_new)])
    
    ## Here we store the results in a data frame
    NN_results = pd.DataFrame(columns = ['number_of_neurons',
                                         'activation',
                                         'activation2',
                                         'cut_off',
                                         'accuracy',
                                         'recall'],
                             data = list_results)
    
    ## Here we compute the harmonic average of the model 
    NN_results['Performance'] = 2/(1/NN_results['accuracy'] + 1/NN_results['recall']) 
    
    ## Here we return the data set of results from this function
    return NN_results

#########
## SVM ##
#########
def SupportVectorMachineResults(X_test, X_train, Y_test, Y_train):
    
    ## Empty list to store results
    list_results = []
    
    SvmHyperParameters = hp.SvmHyperParameters()
 
    ## Here we loop through len of parameters
    SVC_n = SvmHyperParameters.shape[0]
    
    print('SVC')
    for i in tqdm(range(0, SVC_n)):
        
        ## Here we generate our model
        SVC_md = SVC(kernel = SvmHyperParameters.loc[i, 'Kernels'], probability = True).fit(X_train, Y_train)
        
        
        ## Here we make predictions on our test dataset
        SVC_preds = SVC_md.predict_proba(X_test)[:,1]
        
        for k in range(0,N):
            
            
            ## Here we loop though different cut offs
            SVC_preds_new = np.where(SVC_preds < cut_off_list[k],0,1)
            
            ## Here we append the different results
            list_results.append([SvmHyperParameters.loc[i,'Kernels'], 
                                 cut_off_list[k],
                                 accuracy_score(Y_test,SVC_preds_new),
                                 recall_score(Y_test, SVC_preds_new)])


    ## Here we store the results in a data frame        
    SVC_results = pd.DataFrame(columns = ['Kernels','cut_off','accuracy','recall'], data = list_results)
    
    ## Here we compute the harmonic average of the model 
    SVC_results['Performance'] = 2/(1/SVC_results['accuracy'] + 1/SVC_results['recall']) 
     
    ## Here we return the data set of results from this function
    return SVC_results

####################
## SVC Best Model ##
####################
def SvcBestModel(SVC_results):
    
    best_model = SVC_results[SVC_results['Performance'] == max(SVC_results['Performance'])]
    
    return best_model

##########################   
## Logistic Regression ##   
##########################

def LogisticRegressionResults(X_test, X_train, Y_test, Y_train):

    ## Empty list to store results
    list_results = []
     
    ## Here we generate our model        
    LR_md = LogisticRegression().fit(X_train,Y_train)
    
    
    ## Here we make predictions on our test dataset
    LR_preds = LR_md.predict_proba(X_test)[:,1]
        
    print('LogisticRegression')  
    for k in tqdm(range(0,N)):   
        

        
        ## Here we loop though different cut offs
        LR_preds_new = np.where(LR_preds < cut_off_list[k],0,1)
        
        ## Here we append the different results
        list_results.append([cut_off_list[k],
                             accuracy_score(Y_test,LR_preds_new),
                             recall_score(Y_test, LR_preds_new)])
     
    ## Here we store the results in a data frame
    LR_results = pd.DataFrame(columns = ['cut_off','accuracy','recall'], data = list_results)
    
    ## Here we compute the harmonic average of the model 
    LR_results['Performance'] = 2/(1/LR_results['accuracy'] + 1/LR_results['recall']) 
    
    ## Here we return the data set of results from this function
    return LR_results

#############################    
## AdaBoost Decision Trees ##
#############################

def AdaBoostDecisionTreesResults(X_test, X_train, Y_test, Y_train, best_model):
    
    ## Empty list to store results
    list_results = [] 
    
    ADA_parameters = hp.AdaBoostHyperParameters()

    ## Here we loop through len of parameters
    n = ADA_parameters.shape[0]
    
    print('AdaBoostDecisionTrees')
    for i in tqdm(range(0, n)):    
    
        ## Here we generate our model
        ADA_md = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = best_model['max_depth'].reset_index(drop = True)[0]), n_estimators = ADA_parameters.loc[i, 'estimators'], learning_rate = ADA_parameters.loc[i, 'learning_rate']).fit(X_train, Y_train)
        
        ## Here we make predictions on our test dataset
        ADA_preds = ADA_md.predict_proba(X_test)[:,1]  
        
        for k in range(0,N):
            
            ADA_preds_new = np.where(ADA_preds < cut_off_list[k],0,1)
            
            list_results.append([accuracy_score(Y_test,ADA_preds_new),
                                recall_score(Y_test, ADA_preds_new),
                                ADA_parameters.loc[i, 'learning_rate'],
                                ADA_parameters.loc[i, 'estimators'],
                                cut_off_list[k]])
     
    ## Here we store the results in a data frame
    ADA_results = pd.DataFrame(columns = ['accuracy','recall','learning_rate','estimators','cut_off'], data = list_results)
    
    ## Here we store the depth from the best model
    ADA_results['max_depth'] = best_model['max_depth'].reset_index(drop = True)[0]
    
    ## Here we compute the harmonic average of the model 
    ADA_results['Performance'] = 2/(1/ADA_results['accuracy'] + 1/ADA_results['recall']) 
     
    ## Here we return the data set of results from this function
    return ADA_results

###################
## AdaBoost SVM ##
##################

def AdaBoostSvmResults(X_test, X_train, Y_test, Y_train, best_model):
    
    ## Empty list to store results
    list_results = []
    
    ADA_parameters = hp.AdaBoostHyperParameters()
    
    ## Here we loop through len of parameters    
    n = ADA_parameters.shape[0]
    
    print('AdaBoostSVC')
    for i in tqdm(range(0, n)):    
    
        ## Here we generate our model
        ADA_md = AdaBoostClassifier(base_estimator = SVC(kernel = best_model['Kernels'].reset_index(drop = True)[0], probability = True), n_estimators = ADA_parameters.loc[i, 'estimators'], learning_rate = ADA_parameters.loc[i, 'learning_rate']).fit(X_train, Y_train)
        
        ## Here we make predictions on our test dataset
        ADA_preds = ADA_md.predict_proba(X_test)[:,1]
        
        for k in range(0,N):
           
            ## Here we loop though different cut offs
            ADA_preds_new = np.where(ADA_preds < cut_off_list[k],0,1)
            
            ## Here we append the different results
            list_results.append([accuracy_score(Y_test,ADA_preds_new),
                                recall_score(Y_test, ADA_preds_new),
                                ADA_parameters.loc[i, 'learning_rate'],
                                ADA_parameters.loc[i, 'estimators'],
                                cut_off_list[k]])
     
    ## Here we store the results in a data frame
    ADA_results = pd.DataFrame(columns = ['accuracy','recall','learning_rate','estimators','cut_off'], data = list_results)
     
    ## Here we store the depth from the best model   
    ADA_results['Kernels'] = best_model['Kernels'].reset_index(drop = True)[0]
    
    ## Here we compute the harmonic average of the model 
    ADA_results['Performance'] = 2/(1/ADA_results['accuracy'] + 1/ADA_results['recall']) 
   
    ## Here we return the data set of results from this function
    return ADA_results

#######################
## Gradient Boosting ##   
#######################
def GradientBoostingResults(X_test, X_train, Y_test, Y_train):
    
    ## Empty list to store results
    list_results = []
    
    GBM_params = hp.GradientBoostingHyperParameters()

    ## Here we loop through len of parameters
    n = GBM_params.shape[0]
    
    print('GradientBoosting')
    for i in tqdm(range(0, n)):   
        
        
        ## Here we generate our model
        GBC_md = GradientBoostingClassifier(max_depth = GBM_params.loc[i,'max_depth'], n_estimators = GBM_params.loc[i,'estimators'], learning_rate = GBM_params.loc[i,'learning_rate']).fit(X_train, Y_train)
        
        ## Here we make predictions on our test dataset
        GBC_preds = GBC_md.predict_proba(X_test)[:,1]
    
        for k in range(0,N):
            

            ## Here we loop though different cut offs
            GBC_preds_new = np.where(GBC_preds < cut_off_list[k],0,1)
            
            ## Here we append the different results
            list_results.append([GBM_params.loc[i, 'max_depth'],
                                 GBM_params.loc[i, 'estimators'],
                                 GBM_params.loc[i, 'learning_rate'],
                                 accuracy_score(Y_test,GBC_preds_new),
                                 recall_score(Y_test, GBC_preds_new),
                                 cut_off_list[k]])
     
    ## Here we store the results in a data frame
    GBC_results = pd.DataFrame(columns = ['max_depth','estimators','learning_rate','accuracy','recall','cut_off'], data = list_results)
    
    ## Here we compute the harmonic average of the model 
    GBC_results['Performance'] = 2/(1/GBC_results['accuracy'] + 1/GBC_results['recall']) 
    
    ## Here we return the data set of results from this function
    return GBC_results
