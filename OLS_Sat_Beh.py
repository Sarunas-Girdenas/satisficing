# OLS Regression with Satisficing Behavior Algorithm

import matplotlib.pyplot as plt
import numpy as np
from random import choice

# Load data

Data = np.genfromtxt('ex1data1.txt', delimiter=',') # shocks

# Define X & Y variables

X = np.array([Data[:,0]])
Y = np.array([Data[:,1]])

# Add 1's to X (intercept)

X = np.concatenate((np.ones([X.shape[1],1]),X.T),axis=1)

# Define choice sets for alpha & beta

alpha_set = np.linspace(0.0, 10, 100)
beta_set  = np.linspace(0.0, 10, 100)

# Initial choices of alpha and beta

params_OLS = np.zeros([2,1]) # container for OLS parameters

params_OLS[0,0] = choice(alpha_set)
params_OLS[1,0] = choice(beta_set)

# Define Loss Function

def loss_f(X,Y,params_OLS):
	prediction = X.dot(params_OLS).flatten()
	sq_err = (prediction - Y) ** 2
	#J = (1.0 / (2*Y.shape[1])) * sq_err.sum()
	J = sq_err.sum()
	return J

J_loss = loss_f(X,Y,params_OLS) # current loss function

J_loss_new = 0 # initialize future loss function

Er = 0.000001 # errors

no_iter = 0

max_iter = 1000

# Satisficing Behavior Algorithm

for t in range(0,200): 
	
	if abs(J_loss - J_loss_new) > Er:

		J_loss = loss_f(X,Y,params_OLS)

		params_OLS[0,0] = choice(alpha_set)
		params_OLS[1,0] = choice(beta_set)

		J_loss_new = loss_f(X,Y,params_OLS)

	else: 
		params_OLS[0,0] = params_OLS[0,0] 
		params_OLS[1,0] = params_OLS[0,0] 



print params_OLS


# Construct the array of predicted values
X3 = np.zeros([Y.shape[1],2])
X3[:,0] = np.ones([Y.shape[1]])
X3[:,1] = np.asarray(range(1,Y.shape[1]+1))
Y_pred = X3.dot(params_OLS)


plt.figure
plt.scatter(X[:,1],Y)
plt.plot(Y_pred,'r')
plt.show()
