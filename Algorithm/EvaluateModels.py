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


####################
## Decision Trees ##
####################
def DecisionTreesResults(X_test, X_train, Y_test, Y_train):

    DTC_importances = []
    DTC_results = pd.DataFrame()
    DTC_n = hp.DecisionTreesHyperParameters().shape[0]

    print('DecisionTrees')
    for i in tqdm(range(0, DTC_n)):
        DTC = DecisionTreeClassifier(max_depth = hp.DecisionTreesHyperParameters().loc[i, 'max_depth']).fit(X_train,Y_train)

        DTC_preds = DTC.predict_proba(X_test)[:,1]
        DTC_preds = np.where(DTC_preds < hp.DecisionTreesHyperParameters().loc[i, 'cut_off'],0,1)

        DTC_results.loc[i, 'max_depth'] = hp.DecisionTreesHyperParameters().loc[i, 'max_depth']
        DTC_results.loc[i, 'cut_off'] = hp.DecisionTreesHyperParameters().loc[i, 'cut_off']
        DTC_results.loc[i, 'accuracy'] = accuracy_score(Y_test,DTC_preds)
        DTC_results.loc[i, 'recall'] = recall_score(Y_test, DTC_preds)
        
    DTC_results['avg'] = ((DTC_results['accuracy'] + DTC_results['recall'])/2)    

    return DTC_results

##############################
## Decision Trees Best Model##
##############################

def DecisionTreesBestModel(DTC_results):
    
    best_model = DTC_results[DTC_results['avg'] == max(DTC_results['avg'])]
    
    return best_model


###################
## Random Forest ##
###################
def RandomForestResults(X_test, X_train, Y_test, Y_train):

    RF_importances = []
    RF_results = pd.DataFrame()
    RF_n = hp.RandomForestHyperParameters().shape[0]

    print('RandomForest')
    for i in tqdm(range(0, RF_n)):
        RF = RandomForestClassifier(max_depth = hp.RandomForestHyperParameters().loc[i, 'max_depth'],
                                   n_estimators =  hp.RandomForestHyperParameters().loc[i, 'n_estimators']).fit(X_train,Y_train)

        RF_preds = RF.predict_proba(X_test)[:,1]
        RF_preds = np.where(RF_preds < hp.RandomForestHyperParameters().loc[i, 'cut_off'],0,1)

        RF_results.loc[i, 'max_depth'] = hp.RandomForestHyperParameters().loc[i, 'max_depth']
        RF_results.loc[i, 'n_estimators'] = hp.RandomForestHyperParameters().loc[i, 'n_estimators']
        RF_results.loc[i, 'cut_off'] = hp.RandomForestHyperParameters().loc[i, 'cut_off']
        RF_results.loc[i, 'accuracy'] = accuracy_score(Y_test,RF_preds)
        RF_results.loc[i, 'recall'] = recall_score(Y_test, RF_preds)
        
    RF_results['avg'] = ((RF_results['accuracy'] + RF_results['recall'])/2)
    
    return RF_results

#############################
## Random Forest Best Model##
#############################

def DecisionTreesBestModel(RF_results):
    
    best_model = RF_results[RF_results['avg'] == max(RF_results['avg'])]
    
    return best_model

#####################
## Neural Networks ##
#####################

def NeuralNetworksResults(X_test, X_train, Y_test, Y_train):
    
    NnInputParameters = hp.NeuralNetworksHyperParameters()
    
    NN_results = pd.DataFrame()
    NN_n = NnInputParameters.shape[0]

    print('NeuralNetworks')
    for i in tqdm(range(0, NN_n)):
        NN_md = tf.keras.models.Sequential([     
      tf.keras.layers.Dense(NnInputParameters.loc[i, 'number_of_neurons'], input_dim = NnInputParameters.loc[i, 'input_dim'], activation = NnInputParameters.loc[i,'activation']),
          
      tf.keras.layers.Dense(NnInputParameters.loc[i, 'number_of_outputs'], activation = NnInputParameters.loc[i, 'activation2'])])
        
        NN_md.compile(optimizer = NnInputParameters.loc[i, 'optimizer'], metrics = ['accuracy'], loss=NnInputParameters.loc[i, 'loss_function'])
    
        NN_md.fit(X_train,tf.keras.utils.to_categorical(Y_train, num_classes = 2), epochs = 100,batch_size = 500,verbose = 0,
            validation_data = (X_test,tf.keras.utils.to_categorical(Y_test,num_classes = 2)))
    
        NN_preds = NN_md.predict(X_test)[:,1]
        NN_preds = np.where(NN_preds < NnInputParameters.loc[i, 'cut_off'], 0, 1)
        
        NN_results.loc[i, 'number_of_neurons'] = NnInputParameters.loc[i, 'number_of_neurons']
        NN_results.loc[i, 'input_dim'] = NnInputParameters.loc[i, 'input_dim']
        NN_results.loc[i, 'activation'] = NnInputParameters.loc[i, 'activation']
        NN_results.loc[i, 'number_of_outputs'] = NnInputParameters.loc[i, 'number_of_outputs']
        NN_results.loc[i, 'activation2'] = NnInputParameters.loc[i, 'activation2']
        NN_results.loc[i, 'cut_off'] = NnInputParameters.loc[i, 'cut_off']
        NN_results.loc[i, 'accuracy'] = accuracy_score(Y_test,NN_preds)
        NN_results.loc[i, 'recall'] = recall_score(Y_test, NN_preds)
        
    NN_results['avg'] = (NeuralNetworkResults['accuracy'] + NeuralNetworkResults['recall'])/2
    
    
    return NN_results

#########
## SVM ##
#########
def SupportVectorMachineResults(X_test, X_train, Y_test, Y_train):
    
    SvmHyperParameters = hp.SvmHyperParameters()
    
    SVC_results = pd.DataFrame()
    SVC_n = SvmHyperParameters.shape[0]
    
    print('SVC')
    for i in tqdm(range(0, SVC_n)):
        SVC_md = SVC(kernel = SvmHyperParameters.loc[i, 'Kernels'], probability = True).fit(X_train, Y_train)
        
        SVC_preds = SVC_md.predict_proba(X_test)[:,1]
        SVC_preds = np.where(SVC_preds < SvmHyperParameters.loc[i, 'cut_off'],0,1)
        
        SVC_results.loc[i, 'Kernels'] = SvmHyperParameters.loc[i,'Kernels']
        SVC_results.loc[i, 'cut_off'] = SvmHyperParameters.loc[i,'cut_off']  
        SVC_results.loc[i, 'accuracy'] = accuracy_score(Y_test,SVC_preds)
        SVC_results.loc[i, 'recall'] = recall_score(Y_test, SVC_preds)
    SVC_results['avg'] = ((SVC_results['accuracy'] + SVC_results['recall'])/2)
                             
    return SVC_results

####################
## SVC Best Model ##
####################
def SvcBestModel(SVC_results):
    
    best_model = SVC_results[SVC_results['avg'] == max(SVC_results['avg'])]
    
    return best_model

##########################   
## Logistic Regression ##   
##########################

def LogisticRegressionResults(X_test, X_train, Y_test, Y_train):
    parameters = hp.LogisticRegressionParameters() 
    LR_results = pd.DataFrame()
    
    n = parameters.shape[0]
     
    print('LogisticRegression')   
    for i in tqdm(range(0, n)):                     
        LR_md = LogisticRegression().fit(X_train,Y_train)                          
        LR_preds = LR_md.predict_proba(X_test)[:,1]
        LR_preds = np.where(LR_preds < parameters[i],0,1)
        
        LR_results.loc[i, 'cut_off'] = parameters[i]
        LR_results.loc[i, 'accuracy'] = accuracy_score(Y_test,LR_preds)
        LR_results.loc[i, 'recall'] = recall_score(Y_test, LR_preds)
    LR_results['avg'] = ((LR_results['accuracy'] + LR_results['recall'])/2)
    
    return LR_results

#############################    
## AdaBoost Decision Trees ##
#############################

def AdaBoostDecisionTreesResults(X_test, X_train, Y_test, Y_train, best_model):
    
    ADA_parameters = hp.AdaBoostHyperParameters()
    ADA_results = pd.DataFrame()
    n = ADA_parameters.shape[0]
    
    print('AdaBoostDecisionTrees')
    for i in tqdm(range(0, n)):    
    
        ADA_md = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = best_model['max_depth'].reset_index(drop = True)[0]), n_estimators = ADA_parameters.loc[i, 'estimators'], learning_rate = ADA_parameters.loc[i, 'learning_rate']).fit(X_train, Y_train)
        
        ADA_preds = ADA_md.predict_proba(X_test)[:,1]
        ADA_preds = np.where(ADA_preds < ADA_parameters.loc[i, 'cut_off'],0,1)

        ADA_results.loc[i, 'accuracy'] = accuracy_score(Y_test,ADA_preds)
        ADA_results.loc[i, 'recall'] = recall_score(Y_test, ADA_preds)
        ADA_results.loc[i, 'learning_rate'] = ADA_parameters.loc[i, 'learning_rate']
        ADA_results.loc[i, 'estimators'] = ADA_parameters.loc[i, 'estimators']
        
    ADA_results['max_depth'] = best_model['max_depth'].reset_index(drop = True)[0]
    ADA_results['avg'] = ((ADA_results['accuracy'] + ADA_results['recall'])/2)
     
    return ADA_results

###################
## AdaBoost SVM ##
##################

def AdaBoostSvmResults(X_test, X_train, Y_test, Y_train, best_model):
    
    ADA_parameters = hp.AdaBoostHyperParameters()
    ADA_results = pd.DataFrame()
    n = ADA_parameters.shape[0]
    
    print('AdaBoostSVC')
    for i in tqdm(range(0, n)):    
    
        ADA_md = AdaBoostClassifier(base_estimator = SVC(kernel = best_model['Kernels'].reset_index(drop = True)[0], probability = True), n_estimators = ADA_parameters.loc[i, 'estimators'], learning_rate = ADA_parameters.loc[i, 'learning_rate']).fit(X_train, Y_train)
        
        ADA_preds = ADA_md.predict_proba(X_test)[:,1]
        ADA_preds = np.where(ADA_preds < ADA_parameters.loc[i, 'cut_off'],0,1)

            
        ADA_results.loc[i, 'accuracy'] = accuracy_score(Y_test,ADA_preds)
        ADA_results.loc[i, 'recall'] = recall_score(Y_test, ADA_preds)
        ADA_results.loc[i, 'learning_rate'] = ADA_parameters.loc[i, 'learning_rate']
        ADA_results.loc[i, 'estimators'] = ADA_parameters.loc[i, 'estimators']
        
    ADA_results['Kernels'] = best_model['Kernels'].reset_index(drop = True)[0]
    ADA_results['avg'] = ((ADA_results['accuracy'] + ADA_results['recall'])/2)
   
    
    return ADA_results

#######################
## Gradient Boosting ##   
#######################
def GradientBoostingResults(X_test, X_train, Y_test, Y_train):
    
    GBM_params = hp.GradientBoostingHyperParameters()
    GBC_results = pd.DataFrame()
    n = GBM_params.shape[0]
    
    print('GradientBoosting')
    for i in tqdm(range(0, n)):   
        
        GBM_params = hp.GradientBoostingHyperParameters()
        
        GBC_md = GradientBoostingClassifier(max_depth = GBM_params.loc[i,'max_depth'], n_estimators = GBM_params.loc[i,'estimators'], learning_rate = GBM_params.loc[i,'learning_rate']).fit(X_train, Y_train)
    
        
        GBC_preds = GBC_md.predict_proba(X_test)[:,1]
        GBC_preds = np.where(GBC_preds < GBM_params.loc[i, 'cut_off'],0,1)
        
        GBC_results.loc[i, 'max_depth'] = GBM_params.loc[i, 'max_depth']
        GBC_results.loc[i, 'estimators'] = GBM_params.loc[i, 'estimators']
        GBC_results.loc[i, 'learning_rate'] = GBM_params.loc[i, 'learning_rate']
        GBC_results.loc[i, 'accuracy'] = accuracy_score(Y_test,GBC_preds)
        GBC_results.loc[i, 'recall'] = recall_score(Y_test, GBC_preds)
        
    GBC_results['avg'] = ((GBC_results['accuracy'] + GBC_results['recall'])/2)
        
    return GBC_results
