# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:36:03 2019

@author: jig
"""
#%%
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

font = {'family':'normal','weight':'normal','size':16}
mpl.rc('font',**font)

n = [i+1 for i in range(4)]
sigma = [.107,.104,.0706,.0563]

plt.plot(n,sigma)
plt.title('$\sigma_n$ vs number of layer-skipping connections')
plt.xlabel('number of layer-skipping connections')
plt.ylabel('$\sigma_n$')