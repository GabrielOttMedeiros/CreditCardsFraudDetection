import pandas as pd
import numpy as np

## Here we create a list with cutoff values for our prediciotns
cut_off_list = np.arange(0.1,0.6,0.05)

####################################
## Random Forest Hyper-parameters ##
####################################

def RandomForestHyperParameters():
    
    RandomForestParameters = pd.DataFrame()
    
    ## Here we create an array of max_depths
    max_depth = np.arange(3,11,1)

    ## Here we create an array of number of trees
    n_estimators = np.arange(200,1200,200)

    
    RandomForesDepth = []
    RandomForesEstimators = []
    
    ## Here we loop through each list to have every possible combination of hyper-parameters 
    for depth in range(0, len(max_depth)):
        for estimators in range(0, len(n_estimators)):
             for cutoffvalues in range(0, len(cut_off_list)):
                    RandomForesDepth.append(max_depth[depth])
                    RandomForesEstimators.append(n_estimators[estimators])

                    
    
    RandomForestParameters['max_depth'] = RandomForesDepth
    RandomForestParameters['n_estimators'] = RandomForesEstimators

    return RandomForestParameters




#####################################
## Decision Trees Hyper-parameters ##
#####################################


def DecisionTreesHyperParameters():
    
    DecisionTreesParameters = pd.DataFrame()
    
    ## Here we create an array of max_depths
            
    DecisionTreesParameters['max_depth'] =  np.arange(3,11,1)

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

    for neurons in range(0, len(number_of_neurons)):

        for act in range(0, len(activation)):

            for act2 in range(0, len(activation2)):

                for opt in range(0, len(optimizer)):

                    for lss in range(0, len(loss)):

                        NeuralNetworks_number_of_neurons.append(number_of_neurons[neurons])

                        NeuralNetworks_activation.append(activation[act])

                        NeuralNetworks_activation2.append(activation2[act2])

                        NeuralNetworks_optimizer.append(optimizer[opt])

                        NeuralNetworks_loss.append(loss[lss])
                                                            



    NeuralNetworkParameters = pd.DataFrame({'number_of_neurons':NeuralNetworks_number_of_neurons,
                      'activation':NeuralNetworks_activation,
                      'activation2':NeuralNetworks_activation2,
                      'optimizer':NeuralNetworks_optimizer,
                      'loss_function':NeuralNetworks_loss})

    return NeuralNetworkParameters

#########
## SVM ##
#########
def SvmHyperParameters():
    
    list_of_kernels = ['rbf','poly']
    
    kernels_to_append = []
    
    
    for kernels in range(0, len(list_of_kernels)):
        for cut_off_values in range(0, len(cut_off_list)):
            kernels_to_append.append(list_of_kernels[kernels])
        
    
    SvmHyperParameters = pd.DataFrame({'Kernels':kernels_to_append})
    
    return SvmHyperParameters

#########################
## Logistic Regression ##
#########################
def LogisticRegressionParameters():
    
    return cut_off_list

def AdaBoostHyperParameters():
    
    learning_rate_list = [0.01,0.1,1,10]
    n_estimators = np.arange(200,1200,200)
    
    lr_to_append = []
    estimators_to_append = []
    

    for lr in range(0, len(learning_rate_list)):
        for est in range(0, len(n_estimators)):

            estimators_to_append.append(n_estimators[est])
            lr_to_append.append(learning_rate_list[lr])
            
    
    ADA_param = pd.DataFrame({'learning_rate':lr_to_append,
                             'estimators':estimators_to_append})
    
    return ADA_param

######################
## GradientBoosting ##
######################
def GradientBoostingHyperParameters():
    
    learning_rate_list = [0.01,0.1,1,10]
    n_estimators = np.arange(200,1200,200)
    max_depth = max_depth = np.arange(3,11,1)
    
    lr_to_append = []
    estimators_to_append = []
    depth_to_append = []

    for lr in range(0, len(learning_rate_list)):
        for est in range(0, len(n_estimators)):
            for depth in range(0, len(max_depth)):

                estimators_to_append.append(n_estimators[est])
                co_to_append.append(cut_off_list[cut_off])
                lr_to_append.append(learning_rate_list[lr])
                depth_to_append.append(max_depth[depth])

            
    
    GBC_param = pd.DataFrame({'learning_rate':lr_to_append,
                             'estimators':estimators_to_append,
                             'max_depth':depth_to_append})
    
    return GBC_param
