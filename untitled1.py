# -*- coding: utf-8 -*-
#%%

import numpy as np
from matplotlib import pyplot as plt

def R(p):
    return 1 / (1 - np.exp(-p))

p = np.linspace(0,140e-6,10000)

plt.plot(p,R(p))