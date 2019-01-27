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

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
