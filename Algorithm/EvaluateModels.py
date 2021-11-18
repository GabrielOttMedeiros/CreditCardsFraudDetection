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

cut_off_list = np.arange(0.1,0.6,0.05)

N = len(cut_off_list)

####################
## Decision Trees ##
####################
def DecisionTreesResults(X_test, X_train, Y_test, Y_train):

    DTC_results = pd.DataFrame()
    DTC_n = hp.DecisionTreesHyperParameters().shape[0]

    print('DecisionTrees')
    for i in tqdm(range(0, DTC_n)):
        DTC = DecisionTreeClassifier(max_depth = hp.DecisionTreesHyperParameters().loc[i, 'max_depth']).fit(X_train,Y_train)
        
        for k in range(0,N):            
            DTC_preds = DTC.predict_proba(X_test)[:,1]
            DTC_preds = np.where(DTC_preds < cut_off_list[k],0,1)

            DTC_results.loc[k, 'max_depth'] = hp.DecisionTreesHyperParameters().loc[i, 'max_depth']
            DTC_results.loc[k, 'cut_off'] = cut_off_list[k]
            DTC_results.loc[k, 'accuracy'] = accuracy_score(Y_test,DTC_preds)
            DTC_results.loc[k, 'recall'] = recall_score(Y_test, DTC_preds)
        
    DTC_results['Performance'] = 2/(1/DTC_results['accuracy'] + 1/DTC_results['recall'])   

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

    RF_importances = []
    RF_results = pd.DataFrame()
    RF_n = hp.RandomForestHyperParameters().shape[0]

    print('RandomForest')
    for i in tqdm(range(0, RF_n)):
        RF = RandomForestClassifier(max_depth = hp.RandomForestHyperParameters().loc[i, 'max_depth'],
                                   n_estimators =  hp.RandomForestHyperParameters().loc[i, 'n_estimators']).fit(X_train,Y_train)
        for k in range(0,N):

            RF_preds = RF.predict_proba(X_test)[:,1]
            RF_preds = np.where(RF_preds < cut_off_list[k],0,1)

            RF_results.loc[k, 'max_depth'] = hp.RandomForestHyperParameters().loc[i, 'max_depth']
            RF_results.loc[k, 'n_estimators'] = hp.RandomForestHyperParameters().loc[i, 'n_estimators']
            RF_results.loc[k, 'cut_off'] = cut_off_list[k]
            RF_results.loc[k, 'accuracy'] = accuracy_score(Y_test,RF_preds)
            RF_results.loc[k, 'recall'] = recall_score(Y_test, RF_preds)
        
    RF_results['Performance'] = 2/(1/RF_results['accuracy'] + 1/RF_results['recall']) 
    
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
    
    NnInputParameters = hp.NeuralNetworksHyperParameters()
    
    number_of_columns = len(X_train.columns)
    NN_results = pd.DataFrame()
    NN_n = NnInputParameters.shape[0]

    print('NeuralNetworks')
    for i in tqdm(range(0, NN_n)):
        NN_md = tf.keras.models.Sequential([     
      tf.keras.layers.Dense(NnInputParameters.loc[i, 'number_of_neurons'], input_dim = number_of_columns, activation = NnInputParameters.loc[i,'activation']),
          
      tf.keras.layers.Dense(2, activation = NnInputParameters.loc[i, 'activation2'])])
        
        NN_md.compile(optimizer = NnInputParameters.loc[i, 'optimizer'], metrics = ['accuracy'], loss=NnInputParameters.loc[i, 'loss_function'])
    
        NN_md.fit(X_train,tf.keras.utils.to_categorical(Y_train, num_classes = 2), epochs = 100,batch_size = 500,verbose = 0,
            validation_data = (X_test,tf.keras.utils.to_categorical(Y_test,num_classes = 2)))
        
        for k in range(0,N):
    
            NN_preds = NN_md.predict(X_test)[:,1]
            NN_preds = np.where(NN_preds < cut_off_list[k], 0, 1)

            NN_results.loc[k, 'number_of_neurons'] = NnInputParameters.loc[i, 'number_of_neurons']
            NN_results.loc[k, 'activation'] = NnInputParameters.loc[i, 'activation']
            NN_results.loc[k, 'number_of_outputs'] = NnInputParameters.loc[i, 'number_of_outputs']
            NN_results.loc[k, 'activation2'] = NnInputParameters.loc[i, 'activation2']
            NN_results.loc[k, 'cut_off'] = cut_off_list[k]
            NN_results.loc[k, 'accuracy'] = accuracy_score(Y_test,NN_preds)
            NN_results.loc[k, 'recall'] = recall_score(Y_test, NN_preds)
        
    NN_results['Performance'] = 2/(1/NN_results['accuracy'] + 1/NN_results['recall']) 
    
    
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
        
        for k in range(0,N):
        
            SVC_preds = SVC_md.predict_proba(X_test)[:,1]
            SVC_preds = np.where(SVC_preds < cut_off_list[k],0,1)

            SVC_results.loc[k, 'Kernels'] = SvmHyperParameters.loc[i,'Kernels']
            SVC_results.loc[k, 'cut_off'] = cut_off_list[k] 
            SVC_results.loc[k, 'accuracy'] = accuracy_score(Y_test,SVC_preds)
            SVC_results.loc[k, 'recall'] = recall_score(Y_test, SVC_preds)
    SVC_results['Performance'] = 2/(1/SVC_results['accuracy'] + 1/SVC_results['recall']) 
                             
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
    parameters = hp.LogisticRegressionParameters() 
    LR_results = pd.DataFrame()
    
    n = parameters.shape[0]
     
    print('LogisticRegression')   
    for i in tqdm(range(0, n)):                     
        LR_md = LogisticRegression().fit(X_train,Y_train)
        
        for k in range(0,N):
            
            LR_preds = LR_md.predict_proba(X_test)[:,1]
            LR_preds = np.where(LR_preds < cut_off_list[k],0,1)

            LR_results.loc[k, 'cut_off'] = parameters[i]
            LR_results.loc[k, 'accuracy'] = accuracy_score(Y_test,LR_preds)
            LR_results.loc[k, 'recall'] = recall_score(Y_test, LR_preds)
    LR_results['Performance'] = 2/(1/LR_results['accuracy'] + 1/LR_results['recall']) 
    
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
        
        for k in range(0,N):
            
            ADA_preds = ADA_md.predict_proba(X_test)[:,1]
            ADA_preds = np.where(ADA_preds < cut_off_list[k],0,1)

            ADA_results.loc[k, 'accuracy'] = accuracy_score(Y_test,ADA_preds)
            ADA_results.loc[k, 'recall'] = recall_score(Y_test, ADA_preds)
            ADA_results.loc[k, 'learning_rate'] = ADA_parameters.loc[i, 'learning_rate']
            ADA_results.loc[k, 'estimators'] = ADA_parameters.loc[i, 'estimators']
            ADA_results.loc[k, 'cut_off'] = cut_off_list[k]
        
    ADA_results['max_depth'] = best_model['max_depth'].reset_index(drop = True)[0]
    ADA_results['Performance'] = 2/(1/ADA_results['accuracy'] + 1/ADA_results['recall']) 
     
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
        
        for k in range(0,N):
            
            ADA_preds = ADA_md.predict_proba(X_test)[:,1]
            ADA_preds = np.where(ADA_preds < cut_off_list[k],0,1)


            ADA_results.loc[k, 'accuracy'] = accuracy_score(Y_test,ADA_preds)
            ADA_results.loc[k, 'recall'] = recall_score(Y_test, ADA_preds)
            ADA_results.loc[k, 'learning_rate'] = ADA_parameters.loc[i, 'learning_rate']
            ADA_results.loc[k, 'estimators'] = ADA_parameters.loc[i, 'estimators']
            ADA_results.loc[k, 'cut_off'] = cut_off_list[k]
        
    ADA_results['Kernels'] = best_model['Kernels'].reset_index(drop = True)[0]
    ADA_results['Performance'] = 2/(1/ADA_results['accuracy'] + 1/ADA_results['recall']) 
   
    
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
    
        for k in range(0,N):
            
            GBC_preds = GBC_md.predict_proba(X_test)[:,1]
            GBC_preds = np.where(GBC_preds < cut_off_list[k],0,1)

            GBC_results.loc[k, 'max_depth'] = GBM_params.loc[i, 'max_depth']
            GBC_results.loc[k, 'estimators'] = GBM_params.loc[i, 'estimators']
            GBC_results.loc[k, 'learning_rate'] = GBM_params.loc[i, 'learning_rate']
            GBC_results.loc[k, 'accuracy'] = accuracy_score(Y_test,GBC_preds)
            GBC_results.loc[k, 'recall'] = recall_score(Y_test, GBC_preds)
            GBC_results.loc[k, 'cut_off'] = cut_off_list[k]
        
    GBC_results['Performance'] = 2/(1/GBC_results['accuracy'] + 1/GBC_results['recall']) 
        
    return GBC_results
