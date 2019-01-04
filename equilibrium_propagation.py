import numpy as np
import itertools
from matplotlib import pyplot as plt

# TODO / things to try:
# - Divide weight matrix by 10
# - Disable state clipping / allow neuron states to be negative (Bengio STDP-compatible allows it!)


#%% Neuron implementation

layer_sizes = [5, 30, 30, 10]
#layer_sizes = [7*7, 50, 10]
layer_indices = np.cumsum([0] + layer_sizes)
num_neurons = sum(layer_sizes)
W = np.zeros([num_neurons,num_neurons])

# Initialize weights matrix
for n in range(len(layer_sizes)-1): # Make weights only exist from one layer to the next
    wll = np.random.randn(layer_sizes[n+1],layer_sizes[n]) # The weight matrix for one layer to the next
    wll *= np.sqrt(2/(layer_sizes[n]))     # Glorot-Bengio (Xavier) weight normalization
#    wll *= 0.2
    i = layer_indices[n+1]
    j = layer_indices[n]
    di = wll.shape[0]
    dj = wll.shape[1]
    W[i:i+di, j:j+dj] = wll
W += W.T # Make weights symmetric
W = np.matrix(W)

# Initialize state matrix
s = np.random.rand(num_neurons)
s = np.matrix(s).T

# Set up indices of neurons for easy access later
ix = list(range(0, layer_indices[1]))
iy = list(range(layer_indices[-2], layer_indices[-1]))
ih = list(range(layer_indices[1], layer_indices[-2]))
ihy = list(range(layer_indices[1], layer_indices[-1]))


def rho(s):
    return np.clip(s,0,1)

def rhoprime(s):
    rp = s*0
    rp[(0<s) & (s<1)] = 1
    return rp

def E(s, W):
    term1 = sum(0.5*s.T*s)
    term2 = sum(-0.5 * rho(s).T @ W @ rho(s))
#    term3 = -np.sum([b[i]*rho(s[i]) for i in range(len(b))])
    return term1 + term2 # + term3

def E_test(s, W): # Written out explicitly to check form of E(s,W)
    term1 = 0
    term2 = 0
    for i in range(W.shape[0]):
        term1 += 0.5*s[i]**2
        for j in range(W.shape[1]):
            if i == j:
                term2 += 0
            else:
                term2 += -0.5*W[i,j]*rho(s[i])*rho(s[j])
#    term3 = -np.sum([b[i]*rho(s[i]) for i in range(len(b))])
    return sum(term1 + term2) # + term3

#def Rs_test(s, W): # Written out explicitly to check form of E(s,W)
Rs = np.zeros(np.array(s).shape)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Rs[i] += sum(W[i,j]*rho(s[j]))
    
    
eps = 1e-3
# Compute free-phase fixed point
states = []
energies = []
for n in range(10000):
    mu = np.matmul(W,rho(s)) - s
    s[ihy] += eps*mu[ihy]
    s[ihy] = np.clip(s[ihy], 0, 1) # Clip state; equivalent to multiplying R(s) by rhoprime(s)
    states.append(np.array(s).flatten().tolist())
    energies.append(E(s,W))

# Plot states
t = np.linspace(0,len(states) * eps, len(states))
fig, ax = plt.subplots(2,1, sharex = True)
ax[0].plot(t, np.array(states)[:,ih],'r')
ax[0].plot(t, np.array(states)[:,ix],'g')
ax[0].plot(t, np.array(states)[:,iy],'b')
ax[0].set_ylabel('State values')

ax[1].plot(t, np.array(energies),'b')
ax[1].set_ylabel('Energy E')
ax[1].set_xlabel('Time (t/tau)')

# Compute free-phase fixed point