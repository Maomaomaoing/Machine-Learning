# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:20:56 2018

@author: MaoChuLin
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import json

def x_to_fi(M, s, x):
    muj = lambda j,M: 4*j/M
    sigmoid = lambda a: 1/(1+np.exp(0-a))
    
    fi = np.array([0.0]*M).reshape([1,-1])
    for xi in x:
        fi_vec = np.array([0.0]*M).reshape([1,-1])
        for j in range(M):
            a = (xi-muj(j,M)) / s
            fi_vec[0,j] = sigmoid(a)
        fi = np.concatenate([fi, fi_vec], axis = 0)
    fi = fi[1:, :]
    return fi

#%%

""" load data, split data """
data = []
with open('1_data.csv', 'r') as f:
    for row in csv.reader(f):
        data.append(row)

data = data[1:]
data = np.array(data).astype('float32')
datasets = [data[:10], data[:15], data[:30], data[:80]]

""" constant """
M = 7
s = 0.1
m0 = 0
beta = 1
s0_1 = 1e-6*np.ones((M, M))

mNs = []
sNs = []

for data in datasets:
    ## find mean vector mN and the covariance matrix sN
    x = data[:, 0]
    t = data[:, 1].reshape([-1, 1])
    fi = np.matrix(x_to_fi(M, s, x))
    sN_1 = np.matrix(s0_1 + beta * fi.T * fi)
    sN = sN_1.I
    mN = beta * sN * fi.T * t
    mNs.append(np.array(mN).tolist())
    sNs.append(np.array(sN).tolist())
    
    ## generate curve from posterior distribution
    plt.figure(figsize=(10,5))
    w = np.random.multivariate_normal(np.array(mN).reshape([-1]), sN, 5)
    for wi,c in zip(w, ['r', 'y', 'g', 'b', 'c']): 
        pred = np.array(np.matmul(fi, wi)).reshape([-1])
        d = sorted(list(zip(x,pred)), key = lambda x:x[0])
        x1,pred = zip(*d)      
        plt.plot(x1, pred, c)
        
    ## plot x, t
    d = sorted(list(zip(data[:, 0],t)), key = lambda x:x[0])
    x,t = zip(*d)
    plt.figure(figsize=(10,5))
    plt.plot(x, t, 'k')

    ## plot the predictive distribution
    pred_mN = []
    pred_var = []
    for fii in fi:
        fii = np.matrix(fii.reshape([-1, 1]))
        pred_mN.append( float(np.matrix(mN).T @ fii) )
        pred_var.append( float((1/beta)* fii.T @ sN @ fii) )
    std = np.sqrt(pred_var)
    
    ## plot mN, std
    d = sorted(list(zip(data[:, 0],pred_mN)), key = lambda x:x[0])
    x,pred_mN = zip(*d)
    plt.figure(figsize=(10,5))
    plt.plot(x, pred_mN, 'r')
    plt.plot(x, pred_mN+std, 'y')
    plt.plot(x, pred_mN-std, 'y')
    
with open('mNs.json', 'w', encoding='utf8') as f:
    json.dump(mNs, f, indent=4)
with open('sNs.json', 'w', encoding='utf8') as f:
    json.dump(sNs, f, indent=4)