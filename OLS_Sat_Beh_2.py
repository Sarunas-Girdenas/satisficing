# OLS Regression with Satisficing Behavior Algorithm. 
# We aim to approximate OLS with satisficing behavior
# The OLS form we are aiming it is y = b*x, without interecept
# this way we would have 1 dimension loss function which is easier

# BUT IT DOES NOT CONVERGE. WHERE IS THE PROBLEM????

import matplotlib.pyplot as plt
import numpy as np
from random import choice

# Load data

Data = np.genfromtxt('ex1data1.txt', delimiter=',') # shocks

# Define X & Y variables

X = np.array([Data[:,0]])
Y = np.array([Data[:,1]])


# Define choice sets for beta

beta_set  = np.linspace(0.0, 10, 100)

# Initial choices of beta

params_OLS = 1.0

# Define Loss Function

def loss_f(X,Y,params_OLS):

	prediction = X.dot(params_OLS).flatten()
	sq_err = (prediction - Y) ** 2
	J = sq_err.sum()
	
	return J

J_loss = loss_f(X,Y,params_OLS) # current loss function

J_loss_new = 0 # initialize future loss function

Er = 10e-10 # errors

no_iter = 0

params_OLS_container = np.zeros([2000])

# Satisficing Behavior Algorithm

for k in range(0,2000):

	for t in range(0,2000):

		params_OLS = 1.0
		J_loss_new = 0 
	
		if abs(J_loss - J_loss_new) > Er:

			J_loss = loss_f(X,Y,params_OLS)

			params_OLS = choice(beta_set)

			J_loss_new = loss_f(X,Y,params_OLS)

		else:

			params_OLS = params_OLS 

	
	params_OLS_container[k] = params_OLS

	print k

print params_OLS


# Construct the array of predicted values
X3 = np.asarray(range(1,Y.shape[1]+1))
Y_pred = X3.dot(params_OLS)


plt.figure
plt.scatter(X,Y)
plt.plot(Y_pred,'r')
plt.show()

plt.figure
plt.plot(params_OLS_container,'-ko')
plt.show()

# Compare with OLS (closed form equations)

OLS_estimate     = np.cov(X,Y)[0,1] / np.var(X)
Sat_beh_estimate = np.mean(params_OLS_container)