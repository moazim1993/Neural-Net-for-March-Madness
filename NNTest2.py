# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:03:51 2016

@author: moazi
"""
import numpy as np       
# If the deriv=True it calculates the derivative of the function, which is used in the error backpropogation
# Otherwise its used as the sigmoid function turning numbers to probabilities
def nonlin(x, deriv=False):
    if(deriv==True):
         return (x*(1-x)) 
    return (1/(1+np.exp(-x)))   
#debug seed np.random.seed(1)

def run_nn (X,Y,hl):
    in_var =len(X.T) #number of input variable, used to change NN size
    out_var = len(Y.T)
    while hl>2:# number of hidden nodes itterates through many sizes
        syn0 = 2*np.random.random((in_var,hl)) - 1  #subtact bias
    # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
        syn1 = 2*np.random.random((hl,out_var)) - 1  
    # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.           
  # Calculate forward through the network.
        for j in range(50000):#training step
        #updates nodes with new syn
            l0 = X
            l1 = nonlin(np.dot(l0, syn0))
            l2 = nonlin(np.dot(l1, syn1))
            # Back propagation of errors using the chain rule. 
            l2_error = Y - l2
            if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
                abs_err = np.mean(np.abs(l2_error))
                print ("Error: " + str(abs_err))
                if abs_err < .11:#set your error rate
                    np.savetxt("SYN0.csv", syn0, delimiter=",")
                    np.savetxt("SYN1.csv", syn1, delimiter=",")
                    np.savetxt("X.csv", X, delimiter=",")
                    np.savetxt("Yhat.csv", l2, delimiter=",")                                
            l2_delta = l2_error*nonlin(l2, deriv=True)        
            l1_error = l2_delta.dot(syn1.T)       
            l1_delta = l1_error * nonlin(l1,deriv=True)
            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
    
        print ("hidden nodes: ",hl) 
        hl=hl-1    
#___________________________________________________MAIN______________________________________________________
#import data from current directory
data = np.genfromtxt('maxmin_test(13).csv', delimiter=',')
data = np.delete(data, (0), axis=0)#remove labels columb
#set Xs and Y
Y= data[:,2:4]
hl= 9 #largest hiden layer

for i in range(4,np.shape(data)[1]-1,2):
    j=i+1
    for k in range (j+1,np.shape(data)[1],2):
        l = k+1
        for m in range(l+1,np.shape(data)[1],2):
            n = m+1
            X_vars=np.c_[data.T[i],data.T[j],data.T[k],data.T[l],data.T[m],data.T[n]]
            ones = np.ones((649,1)) #add bias
            print ("Testing Variable index")
            print (i,j,k,l,m,n)
            X = np.c_[X_vars,ones]
            run_nn (X,Y,hl)