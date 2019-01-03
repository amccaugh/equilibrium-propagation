import numpy as np
import itertools

# s: state vector (real-valued)
# theta: state vector (real-valued)
# v: state vector (real-valued)


layer_sizes = [28*28, 50, 10]

# Setting up weight matrix
num_neurons = sum(layer_sizes)
W = np.empty([num_neurons,num_neurons])


def rho(s):
    return np.clip(s,0,1)

# Eq (1): The Hopfield-ish energy
def E(layers):
    term1 = np.sum(u**2)
    term2 = 0
    for i, j in itertools.product(range(W.size[0]), range(W.size[1])):
        term2 += W[i,j]*rho(u[i])*rho(u[j])
    term2 *= -0.5
    term3 = -np.sum([b[i]*rho(u[i]) for i in range(len(b))])
    return term1 + term2 + term3
    