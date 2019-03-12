import random

import numpy as np
from numpy import genfromtxt
from scipy import linalg

import matplotlib 
import matplotlib.pyplot as plt

#%%

def load_data(path):
    data = genfromtxt(path, delimiter=',')[1:]
    return data

def normalize(data):
    regularized_data = data.T.copy()
    for i, col in enumerate(data.T):
        maxi = col.max()
        mini = col.min()
        for j, element in enumerate(col):
            regularized_data[i,j] = (element-mini)/(maxi-mini)
    return regularized_data.T

def split_train_test(data):
    data_len = data.shape[0]
    
    train = data[:int(data_len*0.9), :3]
    test = data[int(data_len*0.9):, :3]
    
    train_target = data[:int(data_len*0.9), 3].reshape([-1,1])
    test_target = data[int(data_len*0.9):, 3].reshape([-1,1])
    
    return train, test, train_target, test_target       

def dot_power(x, power, d): 
    # x shape = (1,d), p shape = (d,1)
    x = x.reshape([1,d])
    p = x.T 
    
    for _ in range(power-1):
        p = np.dot(p, x)
        p = p.reshape([-1, 1])
        
    return p.reshape([1,-1])

def x_to_fi_matrix(x, d, M):
    
    fi = np.zeros(( 1, sum([pow(d, i) for i in range(M)]) ))
    
    for row in x:
        temp = np.ones((1,1))
        
        for m in range(1,M):   
            temp = np.concatenate((temp, dot_power(row, m, d)), axis = 1)
        fi = np.concatenate((fi, temp), axis = 0)
        
    return fi[1:]
   
def weight(fi, t, l):
    return linalg.pinv((l * np.identity(fi.shape[1])) + fi.T @ fi) @ fi.T @ t

def error_func(x, t, w, l):
    return np.sum(np.power(x @ w - t, 2), axis = 0) * 0.5 + l * 0.5 * w.T @ w

def model(train, test, train_target, test_target, l, M, d):

    print('M = ',M, ', lambda = ', l)
    try:
        fi = np.mat(x_to_fi_matrix(train, d, M))
        w = weight(fi, train_target, l)
    except:
        print('singular matrix')
        return np.nan
    
    """ train error """
    error = error_func(fi, train_target, w, l)
    N = train.shape[0]
    train_error = np.power(2*error/N, 0.5).item((0,0))
       
    """ test error """
    fi = np.mat(x_to_fi_matrix(test, d, M))
    error = error_func(fi, test_target, w, l)
    N = test.shape[0]
    test_error = np.power(2*error/N, 0.5).item((0,0))
        
    return train_error, test_error
  
#%%


""" main """
n_feature = 3
data = load_data('housing.csv')
np.random.shuffle(data)

#data = normalize(data)
train, test, train_target, test_target = split_train_test(data)

""" least square error, lambda = 0 """
""" regularized least square error, lambda = 0.001, 0.1 """
all_train_error, all_test_error = [], []

for l in [0, 0.001, 0.1]:
    train_error, test_error = [], []
    for M_order in range(1,3+1):
        train_err, test_err = model(train, test, train_target, test_target, l, M_order, n_feature)
        print('train error = ', train_err)
        print('test error = ', test_err)
        train_error.append(train_err)
        test_error.append(test_err)    
    all_train_error.append(train_error)
    all_test_error.append(test_error)

#%%

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

""" plot error """
l = 0
plt.figure(figsize=(25,15))
plt.plot(range(1,len(all_train_error[0])+1), all_train_error[0], label = "train error, lambda=%s" % l, linewidth=1.5)
plt.plot(range(1,len(all_test_error[0])+1), all_test_error[0], label = "test error, lambda=%s" % l, linewidth=1.5)
    
plt.grid()
#plt.xticks(np.linspace(1, 3, 3))
#plt.yticks(np.linspace(0.14, 0.245, 40))
plt.xlabel('M order', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.legend(fontsize=20)

#plt.savefig('error1_shuffle.png')
plt.show()    

plt.figure(figsize=(25,15))


lambda_ = ['0', '0.001', '0.1']
for train_error, l in zip(all_train_error, lambda_):
    plt.plot(range(1,len(train_error)+1), train_error, label = "train error, lambda=%s" % l, linewidth=1.5)
for test_error, l in zip(all_test_error, lambda_):
    plt.plot(range(1,len(test_error)+1), test_error, label = "test error, lambda=%s" % l, linewidth=1.5)
    
plt.grid()
#plt.xticks(np.linspace(1, 3, 3))
#plt.yticks(np.linspace(0.14, 0.245, 40))
plt.xlabel('M order', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.legend(fontsize=20)

#plt.savefig('all_error1_shuffle.png')
plt.show()

#%%

""" select important feature """

# two feature
M_order = 3
l = 0
n_feature = 2

for attr in range(3):
    train, test, train_target, test_target = split_train_test(data)
    train, test = np.delete(train, attr, 1), np.delete(test, attr, 1)
    train_err, test_err = model(train, test, train_target, test_target, l, M_order, n_feature)
    print('train error = ', train_err)

# one feature
M_order = 3
l = 0
n_feature = 1

for attr in range(3):
    train, test, train_target, test_target = split_train_test(data)
    train, test = train[:, attr], test[:, attr]
    train_err, test_err = model(train, test, train_target, test_target, l, M_order, n_feature)
    print('train error = ', train_err)
    
#%%
#A = np.array(range(9)).reshape([3,3])
#
#B = A\np.eye(np.size(A))
