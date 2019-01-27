import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


# Inputs
# X - a N x d matrix with each row corresponding to a training example
# y - a N x 1 column vector indicating the labels for each training example
#
# Outputs
# means - A d x k matrix containing learnt means for each of the k classes
# covmat - A single d x d learnt covariance matrix 

# IMPLEMENT THIS METHOD
def ldaLearn(X,y):
    
    k = 5 # 1 through 5
    d = 2

    sortClasses = dict()
    sortClasses[1] = []
    sortClasses[2] = []
    sortClasses[3] = []
    sortClasses[4] = []
    sortClasses[5] = []

    
    for training_example_index in range(len(y)):
        sortClasses[y[training_example_index][0]].append(X[training_example_index])
    
    means = np.zeros([k, d])
    for key in sortClasses: # key will take values 1 through 5
        means[key - 1] = np.mean(sortClasses[key], axis=0)

    # want 2 by 5
    means = means.T
    
    #This is how it's done for QDA, with separate covariance matrices for each classifier:
    #covmats = [0]*5
    #for k_class in sortClasses:
    #    #print(np.cov(sortClasses[k_class]))
    #    #print(k_class - 1)
    #    npa = np.array(sortClasses[k_class])
    #    covmats[k_class - 1] = np.cov(npa.T)
    
    # For LDA, we just want 1 covariance matrix with all classifiers included:
    covmat = np.cov(X.T)
    
    return means,covmat


# Inputs
# X - a N x d matrix with each row corresponding to a training example
# y - a N x 1 column vector indicating the labels for each training example
#
# Outputs
# means - A d x k matrix containing learnt means for each of the k classes
# covmats - A list of k d x d learnt covariance matrices for each of the k classes

# IMPLEMENT THIS METHOD
def qdaLearn(X,y):

    k = 5 # 1 through 5
    d = 2

    sortClasses = dict()
    sortClasses[1] = []
    sortClasses[2] = []
    sortClasses[3] = []
    sortClasses[4] = []
    sortClasses[5] = []

    
    for training_example_index in range(len(y)):
        sortClasses[y[training_example_index][0]].append(X[training_example_index])
    
    means = np.zeros([k, d])
    for key in sortClasses: # key will take values 1 through 5
        means[key - 1] = np.mean(sortClasses[key], axis=0)

    # want 2 by 5
    means = means.T
   
    covmats = [0]*5
    for k_class in sortClasses:
        #print(np.cov(sortClasses[k_class]))
        #print(k_class - 1)
        npa = np.array(sortClasses[k_class])
        covmats[k_class - 1] = np.cov(npa.T)
     
    return means,covmats


# Inputs
# means, covmat - parameters of the LDA model
# means - a 5 x 2 matrix
# covmats - a k by d x d matrix. 5, 2 x 2 matrices
#
# Xtest - a N x d matrix with each row corresponding to a test example
# ytest - a N x 1 column vector indicating the labels for each test example
# Outputs
# acc - A scalar accuracy value
# ypred - N x 1 column vector indicating the predicted labels 
# IMPLEMENT THIS METHOD
def ldaTest(means,covmat,Xtest,ytest):
    
    means = means.T

    N = len(Xtest)    # number of examples
    d = len(Xtest[0]) # number of dimensions

    determinants = np.linalg.det(covmat)
    #print(determinants)
    #print means.shape
    sig_inv = np.linalg.inv(covmat)
   
    ypred = [] 
    correct_predictions = 0
    total_predictions = 0
    
    for test_example in range(N):
        predicted_probabilities = [0]*5
        for classifier in range(5):
            a = (sqrt(2 * pi))**d
            b = pow(determinants, 0.5)
            x_minus_u = np.subtract(Xtest[test_example], means[classifier])
            first_mult = np.dot(x_minus_u.T, sig_inv)
            #print("shape: " + str(first_mult.shape))
            secon_mult = np.dot(first_mult, x_minus_u)
            scalar_val = np.exp(-1 * secon_mult)
            
            predicted_probabilities[classifier] = scalar_val

        max_prob_index = predicted_probabilities.index(max(predicted_probabilities)) + 1 # Add 1 b/c indexes are 0->4 but labels are 1->5
        
        ypred.append(max_prob_index)
        total_predictions += 1
        if max_prob_index == ytest[test_example]:
            correct_predictions += 1

    acc = correct_predictions / float(total_predictions)
    ypred = np.array(ypred)
    return acc,ypred


def qdaTest(means,covmats,Xtest,ytest):
    
    means = means.T

    N = len(Xtest)    # number of examples
    d = len(Xtest[0]) # number of dimensions

    determinants = np.linalg.det(covmats)
    #print(determinants)
    #print means.shape
   
    ypred = []
    correct_predictions = 0
    total_predictions = 0
    
    for test_example in range(N):
        predicted_probabilities = [0]*5
        for classifier in range(5):
            a = (sqrt(2 * pi))**d
            b = pow(determinants[classifier], 0.5)
            x_minus_u = np.subtract(Xtest[test_example], means[classifier])
            sig_inv = np.linalg.inv(covmats[classifier])
            first_mult = np.dot(x_minus_u.T, sig_inv)
            #print("shape: " + str(first_mult.shape))
            secon_mult = np.dot(first_mult, x_minus_u)
            scalar_val = np.exp(-1 * secon_mult)
            
            predicted_probabilities[classifier] = scalar_val
        max_prob_index = predicted_probabilities.index(max(predicted_probabilities)) + 1
        
        ypred.append(max_prob_index)
        total_predictions += 1
        if max_prob_index == ytest[test_example]:
            correct_predictions += 1

    acc = correct_predictions / float(total_predictions)
        
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # Need numpy array, not Python list
    ypred = np.array(ypred)
    return acc,ypred

