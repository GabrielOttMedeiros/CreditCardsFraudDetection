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
    RandomForestCutoffValues = []
    
    ## Here we loop through each list to have every possible combination of hyper-parameters 
    for depth in range(0, len(max_depth)):
        for estimators in range(0, len(n_estimators)):
             for cutoffvalues in range(0, len(cut_off_list)):
                    RandomForesDepth.append(max_depth[depth])
                    RandomForesEstimators.append(n_estimators[estimators])
                    RandomForestCutoffValues.append(cut_off_list[cutoffvalues])
                    
    
    RandomForestParameters['max_depth'] = RandomForesDepth
    RandomForestParameters['n_estimators'] = RandomForesEstimators
    RandomForestParameters['cut_off'] = RandomForestCutoffValues

    return RandomForestParameters




#####################################
## Decision Trees Hyper-parameters ##
#####################################


def DecisionTreesHyperParameters():
    
    DecisionTreesParameters = pd.DataFrame()
    
    ## Here we create an array of max_depths
    max_depth = np.arange(3,11,1)

    ## Here we create empty lists to hold the results
    DecisionTreesParametersDepth = []
    DecisionTreesParametersCutoff= []

    ## Here we loop through each list to have every possible combination of hyper-parameters 
    for depth in range(0, len(max_depth)):
        for cutoffvalues in range(0, len(cut_off_list)):
            DecisionTreesParametersDepth.append(max_depth[depth])
            DecisionTreesParametersCutoff.append(cut_off_list[cutoffvalues])
            
    DecisionTreesParameters['max_depth'] = DecisionTreesParametersDepth
    DecisionTreesParameters['cut_off'] = DecisionTreesParametersCutoff



    return DecisionTreesParameters

#####################
## Neural Networks ##
#####################

def NeuralNetworksHyperParameters():


    ## Number of neurons to use
    number_of_neurons = np.arange(2,11,1)

    ## Number of columns
    input_dim = 5

    ## Activation funciton 1
    activation = ['relu','tanh'] ## What else?

    ## Activation funciton 2
    activation2 = ['softmax'] ## What else?

    ## Optmizer
    optimizer = ['sgd']

    ## Loss funciton
    loss = ['categorical_crossentropy']

    ## Epochs
    epochs = [100]

    ## Batch size
    batch_size = [500]


    NeuralNetworks_number_of_neurons = []
    NeuralNetworks_input_dim = []
    NeuralNetworks_activation = []
    NeuralNetworks_activation2 = []
    NeuralNetworks_optimizer = []
    NeuralNetworks_loss = []
    NeuralNetworks_epochs = []
    NeuralNetworks_batch_size = []
    NeuralNetworks_cut_off = []

    for neurons in range(0, len(number_of_neurons)):

        for act in range(0, len(activation)):

            for act2 in range(0, len(activation2)):

                for opt in range(0, len(optimizer)):

                    for lss in range(0, len(loss)):

                        for epc in range(0, len(epochs)):

                            for batch in range(0, len(batch_size)):

                                for cutoffvalues in range(0, len(cut_off_list)):

                                    NeuralNetworks_number_of_neurons.append(number_of_neurons[neurons])

                                    NeuralNetworks_activation.append(activation[act])

                                    NeuralNetworks_activation2.append(activation2[act2])

                                    NeuralNetworks_optimizer.append(optimizer[opt])

                                    NeuralNetworks_loss.append(loss[lss])

                                    NeuralNetworks_epochs.append(epochs[epc])

                                    NeuralNetworks_batch_size.append(batch_size[batch])

                                    NeuralNetworks_cut_off.append(cut_off_list[cutoffvalues])



    NeuralNetworkParameters = pd.DataFrame({'number_of_neurons':NeuralNetworks_number_of_neurons,
                       'input_dim':input_dim,
                      'activation':NeuralNetworks_activation,
                      'activation2':NeuralNetworks_activation2,
                      'optimizer':NeuralNetworks_optimizer,
                      'loss_function':NeuralNetworks_loss,
                      'epoch':NeuralNetworks_epochs,
                      'batch_size':NeuralNetworks_batch_size,
                      'cut_off':NeuralNetworks_cut_off})

    NeuralNetworkParameters['number_of_outputs'] = 2

    return NeuralNetworkParameters

#########
## SVM ##
#########
def SvmHyperParameters():
    
    list_of_kernels = ['rbf','poly']
    
    kernels_to_append = []
    
    svm_cut_off = []
    
    for kernels in range(0, len(list_of_kernels)):
        for cut_off_values in range(0, len(cut_off_list)):
            kernels_to_append.append(list_of_kernels[kernels])
            svm_cut_off.append(cut_off_list[cut_off_values])
        
    
    SvmHyperParameters = pd.DataFrame({'Kernels':kernels_to_append,
                                      'cut_off':svm_cut_off})
    
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
    co_to_append = []
    estimators_to_append = []
    
    for cut_off in range(0, len(cut_off_list)):
        for lr in range(0, len(learning_rate_list)):
            for est in range(0, len(n_estimators)):
                
                estimators_to_append.append(n_estimators[est])
                co_to_append.append(cut_off_list[cut_off])
                lr_to_append.append(learning_rate_list[lr])
            
    
    ADA_param = pd.DataFrame({'cut_off':co_to_append,
                             'learning_rate':lr_to_append,
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
    co_to_append = []
    estimators_to_append = []
    depth_to_append = []
    
    for cut_off in range(0, len(cut_off_list)):
        for lr in range(0, len(learning_rate_list)):
            for est in range(0, len(n_estimators)):
                for depth in range(0, len(max_depth)):
                
                    estimators_to_append.append(n_estimators[est])
                    co_to_append.append(cut_off_list[cut_off])
                    lr_to_append.append(learning_rate_list[lr])
                    depth_to_append.append(max_depth[depth])
                    
            
    
    GBC_param = pd.DataFrame({'cut_off':co_to_append,
                             'learning_rate':lr_to_append,
                             'estimators':estimators_to_append,
                             'max_depth':depth_to_append})
    
    return GBC_param
