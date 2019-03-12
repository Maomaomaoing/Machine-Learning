# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:27:47 2018

@author: MaoChuLin
"""

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

def x_to_fi(x):
    return x

def fi_to_y(N, K, w, fi):
    y = np.zeros([N,K])   
    for n in range(N):
        total = 0
        for k in range(K):
            y[n,k] = np.array(np.exp(w[k] @ fi[n].T))
            total += y[n,k]
        y[n] /= total
    return y

def Error(N, K, t, y):
    E = 0
    for n in range(N):
        for k in range(K):
            E += (-1) * t[n,k] * np.log(y[n,k])
    return E

def d1_Error(N, K, t, y, fi):
    Ed1 = np.zeros([K, fi.shape[1]])
    for j in range(K):
        total = np.zeros([1, fi.shape[1]])
        for n in range(N):
            total += (y[n,j]-train_target[n,j])*fi[n]
        Ed1[j] = total
    return Ed1

def d2_Error(N, K, y, fi):
    Ed2 =  np.zeros([K, fi.shape[1], fi.shape[1]])
    fi = np.matrix(fi)
    for j in range(K):
        total = np.zeros([fi.shape[1], fi.shape[1]])
        for n in range(N):
            total += y[n,j] * (1 - y[n,j]) * fi[n].T @ fi[n]
        Ed2[j] = total
    return Ed2

def iter_train(epsilon, E, N, K, train_data, train_target):
    
    ## calculate fi, set w to 0
    fi = x_to_fi(train_data)
    w = np.matrix(np.zeros([K, fi.shape[1]]))
    
    ## iteration
    accs = []
    errors = []
    while E > epsilon:
        ## y
        y = fi_to_y(N, K, w, fi)
        acc = calculate_acc(y, train_target)
#        print('Acc', acc)
        
        ## error
        E = Error(N, K, train_target, y)
        print('Error', E)        
        Ed1 = d1_Error(N, K, train_target, y, fi)   
        Ed2 = d2_Error(N, K, y, fi)
            
        ## update weight         
        for j in range(K):
            H_1 = np.matrix(Ed2[j]).I
            w[j] = w[j] - H_1 @ Ed1[j]
            
        accs.append(acc)
        errors.append(E)
        
    return accs, errors, w

def calculate_acc(y, t):
    acc = 0
    for yi, ti in zip(y, t):
        if np.argmax(yi) == np.argmax(ti):
            acc += 1
    acc /= t.shape[0]
    return acc

def plot(x, y, xlabel, ylabel):
    plt.figure(figsize = (10,6))
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.plot(x, y)

def plot_dot(x, y, xlabel, ylabel, color, title):
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, y, color)

def class_mean(data, label, K):
    mean_vectors = [] 
    for cl in range(1, K+1):
        mean_vectors.append(np.mean(data[label == cl,], axis=0))
    return mean_vectors

def within_class(data, label, K, D):
    m = data.shape[1]
    S_W = np.zeros((m, m))
    mean_vectors = class_mean(data, label, K)
    for cl, mv in zip(range(1, K+1), mean_vectors):
        class_sc_mat = np.zeros((m, m))
        for row in data[label == cl]:
            row, mv = row.reshape(D, 1), mv.reshape(D, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W +=class_sc_mat
    return S_W

def between_class(data, label, K, D):
    m = data.shape[1]
    all_mean =np.mean(data, axis = 0)
    S_B = np.zeros((m, m))
    mean_vectors = class_mean(data, label, K)
    for cl, mean_vec in enumerate(mean_vectors):
        n = data[label == cl+1, :].shape[0]
        mean_vec = mean_vec.reshape(D, 1)
        all_mean = all_mean.reshape(D, 1)
        S_B += n * (mean_vec - all_mean).dot((mean_vec - all_mean).T)
    return S_B

#%%
""" Logistic regression """

## load data
train = genfromtxt('train.csv', delimiter=',')
test_data = genfromtxt('test.csv', delimiter=',')
train_target = train[:, 0:3]
train_data = train[:, 3:]

## set constant
epsilon = 0.01 # when error<epsilon, converge
E = 1000
N = train_data.shape[0] # N data
D = train_data.shape[1] # D feature
K = train_target.shape[1] # K class

accs, errors, w = iter_train(epsilon, E, N, K, train_data, train_target)

plot(range(len(accs)), accs, 'Epoch', 'Accuracy')
plot(range(len(errors)), errors, 'Epoch', 'Error')

#%%
""" classification result """
N = test_data.shape[0] # N data
fi = x_to_fi(test_data)
test_target = fi_to_y(N, K, w, fi)

#%%
""" plot the distribution """
N = train_data.shape[0] # N data
for attr in range(D):
    
    attr_value = [[] for _ in range(K)]   
    for n in range(N):
        attr_value[np.argmax(train_target[n])].append(train_data[n][attr])
        
    plt.figure(figsize = (10,6))
    for j, c in zip(range(K), ['ro', 'yo', 'bo']):
        plot_dot(range(len(attr_value[j])), attr_value[j], '', 'Attribute value', c, 'Feature '+str(attr+1))
## feature 1, 2, 5     

#%%
""" most contributive variables """
## choose feature 1, 5
attr_value = [[] for _ in range(K)]
for n in range(N):
    attr_value[np.argmax(train_target[n])].append([train_data[n][0], train_data[n][4]])
    
plt.figure(figsize = (10,6))
plt.xlabel('Feature 1')
plt.ylabel('Feature 5')
for j, c in zip(range(K), ['ro', 'yo', 'bo']):
    attr1, attr5 = zip(*attr_value[j])
    plt.plot(attr1, attr5, c)

#%%   
""" train by contributive variables """
## choose feature 1, 5
train_data = np.concatenate([train[:, 3].reshape([-1,1]), train[:, 7].reshape([-1,1])], axis = 1)
epsilon = 65.45
accs, errors, w = iter_train(epsilon, E, N, K, train_data, train_target)

plot(range(len(accs)), accs, 'Epoch', 'Accuracy')
plot(range(len(errors)), errors, 'Epoch', 'Error')

## test
test_data = np.concatenate([test_data[:, 0].reshape([-1,1]), test_data[:, 4].reshape([-1,1])], axis = 1)
N = test_data.shape[0] # N data
fi = x_to_fi(test_data)
test_target = fi_to_y(N, K, w, fi)

#%%
""" Fisher’s linear discriminant  """
## load data
K = 3 # class
D = 7 # data dimension

train = genfromtxt('train.csv', delimiter=',')
train_target = train[:, 0:3]
train_data = train[:, 3:]
train_target = np.array([np.argmax(target)+1 for target in train_target])

## compute SW, SB, eigen value and vector of J
S_W = within_class(train_data, train_target, K, D)
S_B = between_class(train_data, train_target, K, D)
eig_vals, eig_vecs = np.linalg.eig( np.linalg.inv(S_W) * S_B )

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(D, 1)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
W = np.hstack((eig_pairs[0][1].reshape(D, 1), eig_pairs[1][1].reshape(D, 1)))

## dimension 7 -> 2
new_data = np.zeros([2])
for data in train_data:
    new_data = np.vstack((new_data, data @ W))
new_data = new_data[1:]
feature1, feature2 = new_data[:, 0], new_data[:, 1]

## plot
plt.figure(figsize = (10,6))
plot_dot(feature1, feature2, 'Feature 1', 'Feature 2', 'o', 'Fisher’s linear discriminant')
