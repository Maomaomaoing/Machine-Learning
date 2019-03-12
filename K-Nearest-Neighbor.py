# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:42:26 2018

@author: MaoChuLin
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    total = 0
    for x1i, x2i in zip(x1, x2):
        total += (x1i-x2i)**2
    return np.sqrt(total)

def K_neighbor(test_data, train_data, K, train_target):
    distance = []
    for data in train_data:
        distance.append(euclidean_distance(test_data, data))
    neighbors = np.argsort(distance)[:K]
    count_class = Counter([train_target[k] for k in neighbors])
    test_target = int(count_class.most_common()[0][0])
    return test_target

def V_neighbor(test_data, train_data, K, train_target, V):
    distance = []
    for data in train_data:
        distance.append(euclidean_distance(test_data, data))
    distance = [d if d<V else -1 for d in distance]
    neighbors = [i for i,d in enumerate(distance) if d != -1]
    count_class = Counter([train_target[k] for k in neighbors])
    test_target = int(count_class.most_common()[0][0])
    return test_target

def plot(x, y, xlabel, ylabel, title):
    plt.figure(figsize = (10,6))
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, y)
    
    
#%%

""" KNN Classifier """
#""" load file """
data = np.genfromtxt('seeds.csv', delimiter = ',')
data = data[1:]
data, target = data[:, :7], data[:, 7]

train_data = np.concatenate([data[0:50], data[70:120], data[140:190]], axis = 0)
train_target = np.concatenate([target[0:50], target[70:120], target[140:190]], axis = 0)
test_data = np.concatenate([data[50:70], data[120:140], data[190:]], axis = 0)
test_target = np.concatenate([target[50:70], target[120:140], target[190:]], axis = 0)

#""" normalization """
for f, attr in enumerate(train_data.T):
    mean = np.mean(attr)
    sd = np.sqrt(np.var(attr))
    train_data[:,f] = ((attr - mean) / sd).T

for f, attr in enumerate(test_data.T):
    mean = np.mean(attr)
    sd = np.sqrt(np.var(attr))
    test_data[:,f] = ((attr - mean) / sd).T

#""" test """
accs = []
for K in range(1,11):
    acc = 0      
    for data, target in zip(test_data, test_target):
        predict = K_neighbor(data, train_data, K, train_target)
        if target == predict:
            acc += 1
    acc /= len(test_data)
    accs.append(acc)
    print('acc:', acc)

#""" plot """
plot(range(1,11), accs, 'K', 'Accuracy', 'K-Nearest-Neighbor Classifier')

#%%
""" Another solution """
#""" test """
accs = []

for V in range(2,11):
    acc = 0      
    for data, target in zip(test_data, test_target):
        predict = V_neighbor(data, train_data, K, train_target, V)
        if target == predict:
            acc += 1
    acc /= len(test_data)
    accs.append(acc)
    print('acc:', acc)

#""" plot """
plot(range(2,11), accs, 'K', 'Accuracy', 'K-Nearest-Neighbor Classifier')
