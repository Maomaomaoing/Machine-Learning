# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:32:04 2018

@author: MaoChuLin
"""
import json
import numpy as np
import matplotlib.pyplot as plt

delta_func = lambda x1, x2: 1 if x1 == x2 else 0
scalar_c = lambda x, beta_1, thetas: kernel_func(x,x, thetas) + beta_1
vector_k = lambda x, xn1, N, thetas: np.array([kernel_func(x[n], xn1, thetas) \
                                    for n in range(N)]).reshape(-1,1)
err_rms = lambda m, t, N: np.sqrt(sum(np.power(m - t, 2)) / N)

#def kernel_func(xn, xm, thetas):
#    dis2 = lambda x1, x2: sum([(xi-xj)**2 for xi, xj in zip(x1,x2)])
#    t0, t1, t2, t3 = thetas
#    res = t0 * np.exp( -t1*dis2(xn, xm)/2 ) + t2 + t3 * xn.T @ xm
#    return res

def kernel_func(xn, xm, thetas):
    dis2 = lambda x1, x2: sum([(xi-xj)**2 for xi, xj in zip([x1],[x2])])
    t0, t1, t2, t3 = thetas
    res = t0 * np.exp( -t1*dis2(xn, xm)/2 ) + t2 + t3 * xn.T * xm
    return res


def cov_mat(x, n, m, thetas, beta_1):
    xn, xm = x[n], x[m]
    res = kernel_func(xn, xm, thetas) + beta_1 * delta_func(n, m)   
    return res

def pred_mean_cov(x, xn1, t, N, thetas, beta_1):    
    ## vector k
    k = vector_k(x, xn1, N, thetas)
    ## covariance matrix
    CN = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            CN[i, j] = cov_mat(x, i, j, thetas, beta_1)
    CN_1 = np.mat(CN).I
    ## t
    t = t.reshape(-1,1)
    ## scalar c
    c = scalar_c(xn1, beta_1, thetas)
    ## mean and covariance
    mean = k.T @ CN_1 @ t
    cov = c - k.T @ CN_1 @ k
    
    return mean, cov 
#%%
""" load file """
with open("gp.csv", 'r') as f:
    data = np.genfromtxt(f, delimiter = ',')

data = np.array(sorted(data, key = lambda x: x[0]))

target, data = data[:,1], data[:, 0]
train_target, test_target, train_data, test_data = target[:60], target[60:], data[:60], data[60:]

N = len(train_data)
beta_1 = 1

""" hyperparameter theta """
different_thetas = [[1, 4, 0, 0], 
                    [0, 0, 0, 1],
                    [1, 4, 0, 5],
                    [1, 64, 10, 0]]
""" plot prediction result """
for thetas in different_thetas:
    means, covs = [], []
    for data in train_data:
        mean, cov = pred_mean_cov(train_data, data, train_target,
                                  N, thetas, beta_1)
        means.append(mean.item()); covs.append(cov.item())
    means, covs = np.array(means), np.array(covs)
    
    plt.figure(figsize=(10,6))
    plt.title('{' + ', '.join(map(str, thetas)) + '}')
    plt.plot(range(N), means, 'r')
    plt.plot(range(N), means + covs, 'y')
    plt.plot(range(N), means - covs, 'y')
    plt.plot(range(N), train_target, 'o')
        
""" calculate root-mean-square errors """
train_err, test_err = [], []
for thetas in different_thetas:
    
    ## train rms error
    means = []
    for data in train_data:
        mean, _ = pred_mean_cov(train_data, data, train_target,
                                  N, thetas, beta_1)
        means.append(mean.item())
    means = np.array(means)
    err = err_rms(means, train_target, N)
    train_err.append(err)
    
    ## test rms error
    means = []
    for data in test_data:
        mean, _ = pred_mean_cov(train_data, data, train_target,
                                  N, thetas, beta_1)
        means.append(mean.item())
    means = np.array(means)
    err = err_rms(means, test_target, N)
    test_err.append(err)

## save file
with open("train_err.json", "w") as f:
    json.dump(train_err, f, indent = 4)
with open("test_err.json", "w") as f:
    json.dump(test_err, f, indent = 4)
