import numpy as np
import itertools
from matplotlib import pyplot as plt

# TODO / things to try:
# - Divide weight matrix by 10
# - Disable state clipping / allow neuron states to be negative (Bengio STDP-compatible allows it!)
# - Try small-world style connections (~N/2^k connections per layer to layers k distance away)


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
    rp[(0<=s) & (s<=1)] = 1 # SUPER IMPORTANT! if (0<s) & (s<1) zeros + ones cannot escape
    return rp

def E(s, W):
    term1 = 0.5*s.T*s
    term2 = -0.5 * rho(s).T @ W @ rho(s)
#    term3 = -np.sum([b[i]*rho(s[i]) for i in range(len(b))])
    return sum(term1 + term2) # + term3

def C(y, d):
    return 0.5*np.linalg.norm(y-d)**2

def F(s, W, beta, d):
    if beta == 0:
        return E(s, W)
    return sum(E(s, W) + beta*C(y = s[iy], d = d)) # + term3

#def step(s, W, beta, d):
#    Rs = np.matmul(W,rho(s))
#    s[ihy] += eps*(Rs - s)[ihy] # dE/ds term
#    if beta != 0:
#        s[iy]  += eps*beta*(d - s[iy]) # beta*dC/ds term
#    # Clipping prevents states from becoming negative due to bad (Euler) time integration
#    # Also, clipping = equivalent to multiplying R(s) by rhoprime(s) when beta = 0
#    s[ihy] = np.clip(s[ihy], 0, 1)  
#    return s
    

eps = 1e-1
# Compute free-phase fixed point
states = []
energies = []
for n in range(10000):
    Rs = np.matmul(W,rho(s))
    mu = Rs - s
    s[ihy] += eps*mu[ihy] # Update output and hidden neurons
    # Clipping prevents states from becoming negative due to Euler integration
    # Also, clipping = equivalent to multiplying R(s) by rhoprime(s) when beta = 0
    s[ihy] = np.clip(s[ihy], 0, 1)  
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

# Compute weakly-clamped fixed point
states = []
energies = []
beta = 0.1
d = np.matrix(np.zeros(len(iy))).T # Target
d[3] = 0.5
for n in range(1000):
    Rs = np.multiply(np.matmul(W,rho(s)), rhoprime(s))
#    Rs[iy] += beta*(s[iy] - d)
#    dEds = -(np.multiply(np.matmul(W,rho(s)), rhoprime(s)) - s)
#    dCds = -beta*(d- s[iy])
#    dsdt = -dEds - beta*dCds
    mu = Rs - s
    s[ihy] += eps*(Rs - s)[ihy] # Update output and hidden neurons
    s[iy]  += eps*beta*(d - s[iy])
    # Clipping prevents states from becoming negative due to bad (Euler) time integration
    # Also, clipping = equivalent to multiplying R(s) by rhoprime(s) when beta = 0
    s[ihy] = np.clip(s[ihy], 0, 1)  
    states.append(np.array(s).flatten().tolist())
    energies.append(F(s, W, beta, d))
    
# Plot states
t = np.linspace(0,len(states) * eps, len(states))
fig, ax = plt.subplots(2,1, sharex = True)
ax[0].plot(t, np.array(states)[:,ih],'r')
ax[0].plot(t, np.array(states)[:,ix],'g')
ax[0].plot(t, np.array(states)[:,iy],'b')
ax[0].set_ylabel('State values')

ax[1].plot(t, np.array(energies),'b')
ax[1].set_ylabel('Energy F')
ax[1].set_xlabel('Time (t/tau)')