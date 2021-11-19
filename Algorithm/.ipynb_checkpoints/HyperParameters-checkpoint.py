import pandas as pd
import numpy as np

####################################
## Random Forest Hyper-parameters ##
####################################

def RandomForestHyperParameters():
    
    RandomForestParameters = pd.DataFrame()
    
    ## Here we create an array of max_depths
    max_depth = np.arange(3,11,1)

    ## Here we create an array of number of trees
    n_estimators = np.arange(200,1200,200)

    ## Here we create an empty lists to hold al combinations of variables 
    RandomForesDepth = []
    RandomForesEstimators = []
    
    ## Here we loop through each list to have every possible combination of hyper-parameters 
    for depth in range(0, len(max_depth)):
        for estimators in range(0, len(n_estimators)):
            
            ## Here we store the resutls
            RandomForesDepth.append(max_depth[depth])
            RandomForesEstimators.append(n_estimators[estimators])                 
    
    ## Here we define a column to hold the multiple hyper parameter combinations
    RandomForestParameters['max_depth'] = RandomForesDepth
    RandomForestParameters['n_estimators'] = RandomForesEstimators

    ## Here we return the DataFrame with our hyper parameter combinations
    return RandomForestParameters




#####################################
## Decision Trees Hyper-parameters ##
#####################################


def DecisionTreesHyperParameters():
    
    ## Here we create an empty dataframe to hold the results
    DecisionTreesParameters = pd.DataFrame()
    
    ## Here we create attach an array with all depths for our data frame
    DecisionTreesParameters['max_depth'] =  np.arange(3,11,1)

    ## Here we return the DataFrame with our hyper parameter combinations
    return DecisionTreesParameters

#####################
## Neural Networks ##
#####################

def NeuralNetworksHyperParameters():


    ## Number of neurons to use
    number_of_neurons = np.arange(2,11,1)
    
    ## Activation funciton 1
    activation = ['relu','tanh'] ## What else?

    ## Activation funciton 2
    activation2 = ['softmax'] ## What else?

    ## Optmizer
    optimizer = ['sgd']

    ## Loss funciton
    loss = ['categorical_crossentropy']

    NeuralNetworks_number_of_neurons = []
    NeuralNetworks_activation = []
    NeuralNetworks_activation2 = []
    NeuralNetworks_optimizer = []
    NeuralNetworks_loss = []

    ## Here we loop through our lists 
    for neurons in range(0, len(number_of_neurons)):

        for act in range(0, len(activation)):

            for act2 in range(0, len(activation2)):

                for opt in range(0, len(optimizer)):

                    for lss in range(0, len(loss)):

                        ## Here we append all combinations to a list
                        NeuralNetworks_number_of_neurons.append(number_of_neurons[neurons])

                        NeuralNetworks_activation.append(activation[act])

                        NeuralNetworks_activation2.append(activation2[act2])

                        NeuralNetworks_optimizer.append(optimizer[opt])

                        NeuralNetworks_loss.append(loss[lss])
                                                            


    ## Here we generate a Data Frame with all hyper parameters combinations
    NeuralNetworkParameters = pd.DataFrame({'number_of_neurons':NeuralNetworks_number_of_neurons,
                      'activation':NeuralNetworks_activation,
                      'activation2':NeuralNetworks_activation2,
                      'optimizer':NeuralNetworks_optimizer,
                      'loss_function':NeuralNetworks_loss})

    ## Here we return the DataFrame with our hyper parameter combinations
    return NeuralNetworkParameters

#########
## SVM ##
#########
def SvmHyperParameters():
    
    ## Here we list the kernels we want to use
    list_of_kernels = ['rbf','poly']
    
    ## Here we crete an empty list to store the results
    kernels_to_append = []
    
    ## ere we loop though our list of parameters and append it to a data frame
    for kernels in range(0, len(list_of_kernels)):
        kernels_to_append.append(list_of_kernels[kernels])
        
    ## Here we create a data frmae with the results
    SvmHyperParameters = pd.DataFrame({'Kernels':kernels_to_append})
    
    ## Here we return the DataFrame with our hyper parameter combinations
    return SvmHyperParameters

#########################
## Logistic Regression ##
#########################
def LogisticRegressionParameters():
    
    ## Here we return a list of cut offs
    return cut_off_list

def AdaBoostHyperParameters():
    
    ## Here we define a range for our hyper parameters
    learning_rate_list = [0.01,0.1,1,10]
    n_estimators = np.arange(200,1200,200)
    
    ## Here we create empty lists to append the reuslts
    lr_to_append = []
    estimators_to_append = []

    ## Here we loop through all combinations and append to our lists
    for lr in range(0, len(learning_rate_list)):
        for est in range(0, len(n_estimators)):
            estimators_to_append.append(n_estimators[est])
            lr_to_append.append(learning_rate_list[lr])
            
    ## Here we create a data frame wih our results
    ADA_param = pd.DataFrame({'learning_rate':lr_to_append,
                             'estimators':estimators_to_append})
    
    ## Here we return the DataFrame with our hyper parameter combinations
    return ADA_param

######################
## GradientBoosting ##
######################
def GradientBoostingHyperParameters():
    
    ## Here we define a range for our hyper parameters
    learning_rate_list = [0.01,0.1,1,10]
    n_estimators = np.arange(200,1200,200)
    max_depth = max_depth = np.arange(3,11,1)
    
    ## Here we create empty lists to append the reuslts
    lr_to_append = []
    estimators_to_append = []
    depth_to_append = []

    ## Here we loop through all combinations and append to our lists
    for lr in range(0, len(learning_rate_list)):
        for est in range(0, len(n_estimators)):
            for depth in range(0, len(max_depth)):

                estimators_to_append.append(n_estimators[est])
                lr_to_append.append(learning_rate_list[lr])
                depth_to_append.append(max_depth[depth])

            
    ## Here we create a data frame wih our results
    GBC_param = pd.DataFrame({'learning_rate':lr_to_append,
                             'estimators':estimators_to_append,
                             'max_depth':depth_to_append})
    
    ## Here we return the DataFrame with our hyper parameter combinations
    return GBC_param