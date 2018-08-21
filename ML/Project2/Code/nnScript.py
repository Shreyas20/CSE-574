import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import time
import pickle
def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1.0 / (1.0 + np.exp(-z))
def preprocess():
    mat = loadmat('mnist_all.mat')
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]
    

    # Feature selection
    # Your code here.
    std=np.std(train_data,0)
    included_features=[]
    removed_features=[]
    for i in range(train_data.shape[1]):
        if (std[i]>0.002):
            included_features.append(i)
        else:
            removed_features.append(i)
    train_data = np.delete(train_data, removed_features,1)
    validation_data = np.delete(validation_data,removed_features,1)
    test_data = np.delete(test_data, removed_features,1)
    #train_label=np.reshape(train_label,(len(train_label),1))
    #test_label=np.reshape(test_label,(len(test_label),1))
    #validation_label=np.reshape(validation_label,(len(validation_label),1))
    
    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label,included_features
def onehotvector(training_label,n_class):
    new_label=np.zeros((training_label.shape[0],n_class),dtype=np.int)
    for i in range(training_label.shape[0]):
        for index in range(n_class):
            if(index==int(training_label[i])):
            
             new_label[i][index]=1
    
    return new_label
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    obj_grad = np.array([])
    n=(training_data.shape[0])
    b = np.ones((n, 1))

    X = np.concatenate((training_data,b), axis = 1)

    
    A=np.dot(w1,X.T)
    
    Z=sigmoid(A)
    Z=Z.T
    Z=np.array(Z)
    
    #n=np.shape(t)[0]
    b1 = np.ones((len(Z), 1))

    Z = np.concatenate((Z,b1), axis = 1)

#     for i in range(len(Z)):
#         Z[i].append(1)
#     Z=Z
        
    B=np.dot(w2,Z.T)
    
    O=sigmoid(B)
    #print(O[2])
    Y=onehotvector(training_label,n_class)
    Y=Y.T
    #print(Y.shape)
    Error_Function=Y*np.log(O) + (1 - Y)*np.log(1 - O) 
    EF=-(1/n)*(np.sum(Error_Function[:,:]))
    delta_output=O-Y
    w2err=np.dot(delta_output,Z)
    #print(w2err)
    delta_hidden=np.dot(w2.T,delta_output)*((Z.T)*(1-Z.T))

    #print(delta_output[1:9])
  #  w2err=np.dot(delta_output,Z)
  #  delta_hidden=np.dot(w2.T, delta_output)*(Z.T*(1 - Z.T))
    w1err=np.dot(delta_hidden,X)
    w1err=w1err[:-1,:]
    Regularization=(lambdaval/(2*n))*(np.sum(w1**2)+np.sum(w2**2))
    obj_val=EF+Regularization
   # print(obj_val)
   
    grad_w1=(lambdaval*w1+w1err)/n
    grad_w2=(lambdaval*w2+w2err)/n
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    print(obj_val)

    
    
    # Your code here
    #
    #
    #
    #
    #
    
    

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    

    return (obj_val, obj_grad)
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    n=(data.shape[0])
    b = np.ones((n, 1))
    X = np.concatenate((data,b), axis = 1)
    
    A=np.dot(w1,X.T)
    
    Z=sigmoid(A)
    Z=Z.T
    b1 = np.ones((len(Z), 1))

    Z = np.concatenate((Z,b1), axis = 1)

    B=np.dot(w2,Z.T)
    
    O=sigmoid(B)
    labels=(np.argmax(O,0))
    return labels

train_data, train_label, validation_data, validation_label, test_data, test_label,features = preprocess()
#train_data1=np.array(train_data)
#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = (train_data.shape[1])

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 20

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

ts=time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
te=time.time()

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj=[features,n_hidden,w1,w2,lambdaval]
with open('params.pickle','wb') as f:
    pickle.dump(obj,f)

print("Time required: ",te-ts)
