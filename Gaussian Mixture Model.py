# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:32:04 2018

@author: MaoChuLin
"""
import numpy as np
import random
import json
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_image(filename):
    
#    jpgfile = Image.open(filename)
#    data = np.array(jpgfile)
#    N = data.shape[0]*data.shape[1]*data.shape[2]
#    data = (np.subtract(data, np.array([128]*N).reshape(data.shape))) / 128
#    data = data.reshape(-1, 3)
    data = mpimg.imread('image.jpg').reshape([-1, 3]).astype('float64')
    return data

dis = lambda x, m: sum((x-m)**2)

def init(data, N, K):
    ## random mu, (initial far point)
    mus = [data[random.randint(0, N-1)] for _ in range(K)]
    ## calculate gamma, E step
    gamma = np.zeros([N, K])
    for n in range(N):
        k = np.argmin([ dis(data[n], mus[k]) for k in range(K) ])
        gamma[n, k] = 1
    
    return mus, gamma

def find_mu(data, mus, gamma):
    ## find mu by gamma, M step
    for k in range(K):
        sum1 = np.sum([gamma[n, k]*data[n] for n in range(N)], axis = 0)
        sum2 = np.sum([gamma[n, k] for n in range(N)])
        mus[k] = sum1 / sum2
        
    return mus

def find_gamma(data, mus, gamma):
    ## find gamma by mu, E step
    for n in range(N):
        k = np.argmin([ dis(data[n], mus[k]) for k in range(K) ])
        gamma[n] = np.zeros(gamma[n].shape)
        gamma[n, k] = 1
    
    return gamma

def error_func(data, mus, gamma):   
    ## minimize this error
    error = np.sum([gamma[n, k]*dis(data[n], mus[k]) for n in range(N) for k in range(K)])
    return error

def cov_matrix(x, mu, N):
    x = x.copy()
    x -= mu 
    fact = N - 1 
    cov = np.dot(x.T, x.conj()) / fact
    return cov

def E_step(data, mus, covs, N, K):
    gamma = np.zeros([N, K])
    for n in range(N):
        down = 0
        for j in range(K):
            down += pis[j] * multivariate_normal.pdf(data[n], mean=mus[j], cov=covs[j])
        for k in range(K):
            gamma[n, k] = pis[k] * multivariate_normal.pdf(data[n], mean=mus[k], cov=covs[k]) / down
#        if n % 1000 == 0:    
#            print('E-step:', n)
    return gamma

def M_step(gamma, data, N, K):
    Ns = np.array([sum([gamma[n, k] for n in range(N)]) for k in range(K)])
    pis = Ns / N    
    mus = [ sum([gamma[n, k]*data[n] for n in range(N)]) / Ns[k] for k in range(K) ] 
    for k in range(K):
        total = 0
        for n in range(N):
            total += gamma[n, k] * (data[n]-mus[k]) * (data[n]-mus[k]).T
        covs[k] = total / Ns[k]
    
    return mus, covs, pis
            
def log_likelihood(data, mus, covs, pis):
    total = 0
    for n in range(N):
        temp = 0
        for k in range(K):
            temp += pis[k] * multivariate_normal.pdf(data[n], mean=mus[k], cov=covs[k])
        total += np.log(temp)
    return total

def print_img(K_gamma, K_mus, Ks, method):
    
    jpgfile = mpimg.imread('hw3.jpg')
    img_shape = jpgfile.shape
    
    for gamma, mus, K in zip(K_gamma, K_mus, [2, 3, 5, 20]):
    
        predicts = np.argmax(gamma, axis = 1)
        new_image = np.array([mus[p] for p in predicts]).reshape(img_shape)
        new_image = new_image.astype('uint8')
        
        plt.figure(figsize=(10, 6))
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        plt.title(method + " " + str(K) + " cluster color")
        plt.imshow(new_image)
        plt.savefig(method + "_" + str(K) + '_color.png',bbox_inches='tight')
   
#%%

""" K means as init """

data = read_image('hw3.jpg')
N = data.shape[0]

Ks = [2, 3, 5, 20]
K_mus = []
K_gamma = []
for K in Ks:
    
    print('K =', K)
    error = 1000
    count = 0
    
    mus, gamma = init(data, N, K)
    
    while error > 100:
        mus = find_mu(data, mus, gamma)
        gamma = find_gamma(data, mus, gamma)
        new_error = error_func(data, mus, gamma)
        count += 1
        print("Iter:", count, ", Error:", new_error)
        if np.abs(error - new_error) < 1 or count > 50:
            break
        error = new_error
    
    K_mus.append(mus)
    K_gamma.append(gamma)


K_mus = [[mu.tolist() for mu in mus] for mus in K_mus]  
with open("K_means_mus.json", 'w') as f:
    json.dump(K_mus, f, indent = 4)

K_gamma = [gamma.tolist() for gamma in K_gamma]
with open("K_means_gamma.json", 'w') as f:
    json.dump(K_gamma, f, indent = 4)
    
#%%

""" GMM, EM algorithm """
with open("mus.json", 'r') as f:
    K_mus = json.load(f)
data = read_image('hw3.jpg')
N = data.shape[0]

K_curves = []
K_gamma = []
GMM_mus = []
Ks = [2, 3, 5, 20]
for K, mus in zip(Ks, K_mus):
    print("K =", K)
    ## initialize
    covs = []
    for k in range(K):
        cov = cov_matrix(data, mus[k], N)
        covs.append(cov)
    pis = [1/K] * K
    
    new_mus = mus
    curves = []
    for count in range(100):
        ## E-step
        gamma = E_step(data, new_mus, covs, N, K)       
        ## M-step
        new_mus, covs, pis = M_step(gamma, data, N, K)       
        ## log likelihood
        curve = log_likelihood(data, new_mus, covs, pis)
        curves.append(curve)
        print("Iter:", count,",", curve)
        
    K_curves.append(curves)
    K_gamma.append(gamma)
    GMM_mus.append(new_mus)

K_gamma = [gamma.tolist() for gamma in K_gamma]
GMM_mus = [[mu.tolist() for mu in mus] for mus in GMM_mus]

with open("curves.json", "w") as f:
    json.dump(K_curves, f, indent = 4)
with open("GMM_gamma.json", "w") as f:
    json.dump(K_gamma, f, indent = 4)
with open("GMM_mus.json", "w") as f:
    json.dump(GMM_mus, f, indent = 4)
   
for K, curves in zip(Ks, K_curves):
    plt.figure(figsize=(10, 6))
    plt.title(str(K) + " cluster log likelihood")
    plt.grid()
    plt.plot(range(len(curves)), curves)
    plt.savefig(str(K) + '_cluster.png')

#%%
    
""" different K """

## 1
with open("K_means_gamma.json", "r") as f:
    K_gamma = json.load(f)
with open("K_means_mus.json", "r") as f:
    K_mus = json.load(f)

Ks = [2, 3, 5, 20]

print_img(K_gamma, K_mus, Ks, "K_means")
    
## 2  
with open("GMM_gamma.json", "r") as f:
    K_gamma = json.load(f)
with open("GMM_mus.json", "r") as f:
    K_mus = json.load(f)

Ks = [2, 3, 5, 20]

print_img(K_gamma, K_mus, Ks, "GMM")
 
#%%

""" read table """   
#with open("K_means_mus.json", "r") as f:
#    K_mus = json.load(f)
#
#for i,mus in enumerate(K_mus):
#    for j,mu in enumerate(mus):
#        K_mus[i][j] = np.array(mu).astype('uint8').tolist()
#
#temp = [np.array(mu).reshape((1,1,3)) for mus in K_mus for mu in mus]   
#for i, t in enumerate(temp):
#    plt.figure(figsize=(0.2, 0.2))
#    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
#    plt.imshow(t)