# OLS Regression with Satisficing Behavior Algorithm

import matplotlib.pyplot as plt
from math import ceil
from random import random
import csv

Data = list(csv.reader(open("ex1data1.txt")))    # reading data with csv

X = [float(row[0]) for row in Data]    # I use python list type because they tend to be faster
Y = [float(row[1]) for row in Data]

def loss_f(X,Y,params_OLS):
	prediction = [x*params_OLS[0]+params_OLS[1] for x in X]
	sq_err = [(prediction[i] - Y[i]) ** 2 for i in range(0,len(prediction))]
	return sum(sq_err)


fit = [random(), random()]
J_loss_new = loss_f(X,Y,fit) # current loss
J_loss=0 # container for loss
Er = 0 
Er_correction=0.01 # lowering this value will lead to longer computations but better results

while abs(J_loss - J_loss_new) > Er:
	J_loss = J_loss_new
	fit = [random(), random()] 
	J_loss_new = loss_f(X,Y,fit) # calculate new loss
	Er+=Er_correction   # lower our expectations

plt.figure
plt.scatter(X,Y)
plt.plot([[i*fit[0]+fit[1]] for i in range(0, int(ceil(max(X))))],'r') # our final model
plt.show()
