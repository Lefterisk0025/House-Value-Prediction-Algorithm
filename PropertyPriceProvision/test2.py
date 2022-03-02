from matplotlib.pyplot import axis
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from pandas import *
from scipy import stats

x, y = make_regression(n_samples=20, n_features=9, n_informative=1, 
                       random_state=0, noise=35) 

x1 = np.array([1, 3, 5])
x2 = np.array([5, 2, 8])
xN = np.stack((x1, x2), axis=1)

print(x1)
x1 = x1.reshape(-1, 1)
print(x1)
x1 = np.concatenate(x1, axis=0)
print(x1)

print(x)


