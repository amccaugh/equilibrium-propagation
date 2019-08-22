#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:07:22 2019

@author: jig
"""

layers = [10,10,10,10,10,10,10]


tc = int((sum(layers)**2)/2)
lsc = 0
for i in range(len(layers)):
    lsc += layers[i]*sum(layers[i+2:])
assert tc==.5*(2*lsc+sum([l**2 for l in layers])+2*sum([layers[i]*layers[i+1] for i in range(len(layers)-1)]))

print('{} total connections'.format(tc))
print('{} layer-skipping connections'.format(lsc))