# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:32:04 2018

@author: MaoChuLin
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

linear_fi = lambda x: x
linear_kernel = lambda xi, xj: xi.T @ xj
poly_fi = lambda x: np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])
poly_kernel = lambda xi, xj: poly_fi(xi).T @ poly_fi(xj)

def one_vs_rest(datas, targets):
    ## 0~49, 50~99, 100~149
    ## -1 less
    datasets = []
    for i in range(3):
        targets = np.array([-1]*len(targets))
        targets[i*50:(i+1)*50] = 1
        datasets.append([datas, targets])

    return datasets

def find_coef(clf, N):
    ## find coefficients
    sv_index = clf.support_
    not_sv = set(range(N)) - set(sv_index)
    coef = clf.dual_coef_[0].tolist()
    
    for index in not_sv:
        coef.insert(index, 0)
    return coef, sv_index

def calculate_w_b(coef, sv_index, datas, targets, N, fi, kernel):
    ## find w
    w = sum([coef[n]*targets[n]*linear_fi(datas[n]) for n in range(N)])
    ## find b    
    b = 0
    for n in sv_index:
        atk_sum = 0
        for m in sv_index:
            atk_sum += coef[n]*targets[n]*linear_kernel(datas[n], datas[m])
        b += targets[n] - atk_sum
    b /= len(sv_index)
    
    return w, b

def make_meshgrid(x, y, h=.001):
#    print("Max / Min", x.min(), y.min(), x.max(), y.max())
    x_min, x_max = x.min() - .1, x.max() + .1
    y_min, y_max = y.min() - .1, y.max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot(ws, bs, xx, yy, sv, title):

    ## plot decision boundary
    Z = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        predicts = []
        for w, b in zip(ws, bs):
            predicts.append(w @ linear_fi(np.array([x, y])).T + b)
        predict = np.argmax(np.array(predicts))
        Z.append(predict)
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(15,9))
    plt.title(title)
    plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8)
    
    ## plot data point and support vector
    datas = np.genfromtxt('x_train.csv', delimiter = ',')
    datas1, datas2 = list(zip(*datas))
    vec1, vec2 = list(zip(*sv))
    
    plt.plot(datas1[:50], datas2[:50], 'ro', label = 'Class 1')
    plt.plot(datas1[50:100], datas2[50:100], 'go', label = 'Class 2')
    plt.plot(datas1[100:], datas2[100:], 'bo', label = 'Class 3')        
    plt.plot(vec1, vec2, 'yx', label = 'Support vector') 
    plt.legend(loc='upper right')
    plt.savefig(title)

def calculate_acc(ws, bs):
    datas = np.genfromtxt('x_train.csv', delimiter = ',')
    predicts = np.zeros([N, 3])
    for n, x in enumerate(datas):
        for k, (w, b) in enumerate(zip(ws, bs)):
            y = w.T @ x + b
            predicts[n, k] = y
    
    predicts = np.argmax(predicts, axis = 1)
    acc = ((predicts[:50] == 1).sum() + \
           (predicts[50:100] == 0).sum() + \
           (predicts[100:] == 2).sum()) / N
    return acc
     
#%%
""" load file """
datas = np.genfromtxt('x_train.csv', delimiter = ',')
targets = np.genfromtxt('t_train.csv', delimiter = ',')
N = len(datas)

""" one versus rest """
datasets = one_vs_rest(datas, targets)

""" linear kernel """
## construct 3 model
ws, bs = [], []
for datas, targets in datasets:
    
    ## fit model and find coef
    clf = SVC(kernel='linear')
    clf.fit(datas, targets)
    coef, sv_index = find_coef(clf, N)
    
    ## find support vector
    sv = [datas[index] for index in sv_index]
    ## find w and b
    w, b = calculate_w_b(coef, sv_index, 
                         datas, targets, N, 
                         linear_fi, linear_kernel)
    ws.append(w); bs.append(b)

## plot
xx, yy = make_meshgrid(datas[:, 0], datas[:, 1], h=.02)
plot(ws, bs, xx, yy, sv, "Linear kernel")

## predict
acc = calculate_acc(ws, bs)
print("Linear kernel acc:", acc)

""" polynomial kernel """
## construct 3 model
ws, bs = [], []
for datas, targets in datasets:
    
    ## fit model and find coef
    clf = SVC(kernel='poly', degree = 2)
    clf.fit(datas, targets)
    coef, sv_index = find_coef(clf, N)
    
    ## find support vector
    sv = [datas[index] for index in sv_index]
    
    ## find w and b
    w, b = calculate_w_b(coef, sv_index, 
                         datas, targets, N, 
                         poly_fi, poly_kernel)
    ws.append(w); bs.append(b)

## plot
xx, yy = make_meshgrid(datas[:, 0], datas[:, 1], h=.02)
plot(ws, bs, xx, yy, sv, "Polynomial kernel")

## predict
acc = calculate_acc(ws, bs)
print("Polynomial kernel acc:", acc)