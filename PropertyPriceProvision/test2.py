from tracemalloc import start
from matplotlib.pyplot import axis
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.datasets import make_classification
import numpy as np
import random
from pandas import *
from scipy import stats

def gradient_descent(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    theta = np.ones(x.shape[1])
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        #print ("iter %s | J: %.3f" % (iter, J))      
        gradient = np.dot(x_transpose, loss) / m         
        theta = theta - alpha * gradient  # update
    return theta

x, y = make_regression(n_samples=20, n_features=1, n_informative=1, 
                       random_state=0, noise=35) 

x, y = make_classification(n_samples=100, n_features=9, n_informative=1, random_state=1)  
print(x)    
print(y)                  





