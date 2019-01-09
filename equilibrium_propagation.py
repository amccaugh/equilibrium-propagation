import numpy as np
from matplotlib import pyplot as plt


# TODO / things to try:
# - Divide weight matrix by 10
# - Disable state clipping / allow neuron states to be negative (Bengio STDP-compatible allows it!)
# - Try small-world style W_exists (~N/2^k W_exists per layer to layers k distance away)


#%% Neuron implementation
np.random.seed(seed = 0)


layer_sizes = [12, 25, 10]
layer_indices = np.cumsum([0] + layer_sizes)
num_neurons = sum(layer_sizes)

# Set up indices of neurons for easy access later
ix = list(range(0, layer_indices[1]))
iy = list(range(layer_indices[-2], layer_indices[-1]))
ih = list(range(layer_indices[1], layer_indices[-2]))
ihy = list(range(layer_indices[1], layer_indices[-1]))

def intialize_weight_matrix(layer_sizes, seed = None):
    W = np.zeros([num_neurons,num_neurons])
    W_exists = np.zeros(W.shape)
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
        W_exists[i:i+di, j:j+dj] = wll*0 + 1
    W += W.T # Make weights symmetric
    W_exists += W_exists.T
    W = np.matrix(W)
    W_exists = np.matrix(W_exists)
    return W, W_exists

# Initialize state matrix
def initialize_state(x = None, seed = None):
    np.random.seed(seed = seed)
    s = np.random.rand(num_neurons)
    if x is not None: s[ix] = x
    s = np.matrix(s).T
    return s


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

def step(s, W, eps, beta, d):
    Rs = np.matmul(W,rho(s))
    s[ihy] += eps*(Rs - s)[ihy] # dE/ds term, multiplied by dt (epsilon)
    if beta != 0:
        s[iy]  += eps*beta*(d - s[iy]) # beta*dC/ds weak-clamping term, multiplied by dt (epsilon)
    # Clipping prevents states from becoming negative due to bad (Euler) time integration
    # Also, clipping = equivalent to multiplying R(s) by rhoprime(s) when beta = 0
    s[ihy] = np.clip(s[ihy], 0, 1)  
    return s
    
def evolve_to_equilbrium(s, W, d, beta, eps, total_tau,
                         state_list = None, energy_list = None):
    # If state_recorder is passed an (empty) list, it will append each state to it
    num_steps = int(total_tau/eps)
    for n in range(num_steps):
        step(s, W, eps = eps, beta = beta, d = d)
        if state_list is not None: states.append(np.array(s).flatten().tolist())
        if energy_list is not None: energies.append(F(s, W, beta, d))
    return s
    
def plot_states_and_energy(states, energies):
    # Plot states
    t = np.linspace(0,len(states) * eps, len(states))
    fig, ax = plt.subplots(2,1, sharex = True)
    ax[0].plot(t, np.array(states)[:,ih],'r')
    ax[0].plot(t, np.array(states)[:,ix],'g')
    ax[0].plot(t, np.array(states)[:,iy],'b')
    ax[0].set_ylabel('State values')
    
    ax[1].plot(t, np.array(energies),'b.-')
    ax[1].set_ylabel('Energy E')
    ax[1].set_xlabel('Time (t/tau)')

# Weight update
def weight_update(W, W_exists, beta, s_free_phase, s_clamped_phase):
    # W_exists = matrix of shape(W) with 1s or zeros based on 
    # whether the connection / weight between i and j exists
    dW = 1/beta*(rho(s_clamped_phase) @ rho(s_clamped_phase).T - 
                 rho(s_free_phase) @ rho(s_free_phase).T
                 )
    dW = np.multiply(dW, W_exists)
    return dW


#%% Plot states and energies
seed = 1
eps = 0.01
s = initialize_state(seed = seed)
W,W_exists = intialize_weight_matrix(layer_sizes = layer_sizes, seed = seed)

states = []
energies = []
s = evolve_to_equilbrium(s = s, W = W, d = None, beta = 0, eps = eps, total_tau = 10,
                         state_list = states, energy_list = energies)
s_free_phase = s.copy()

d = np.zeros([10,1])
d[3] = 0.5
s = evolve_to_equilbrium(s = s, W = W, d = d, beta = 1, eps = eps, total_tau = 10,
                         state_list = states, energy_list = energies)
s_clamped_phase = s.copy()
plot_states_and_energy(states, energies)

dW = weight_update(W, W_exists, beta, s_free_phase, s_clamped_phase)


#%% Run algorithm
    
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
# Load mnist data
x_train, t_train, x_test, t_test = load()

# quick
dW_list = []
for n in range(1000):
    # Set parameters
    eps = 0.01
    total_tau = 2
    beta = 1
    
    # Select input
    m = 1
    x = x_train[m,:]
    num = t_train[m]
    d = np.zeros([10,1]); d[num] = 1
    
    # Perform weight update from one sample
    s = initialize_state(x = x)
    W,W_exists = intialize_weight_matrix(layer_sizes = layer_sizes)
    s_free_phase = evolve_to_equilbrium(s = s, W = W, d = None, beta = 0, eps = eps, total_tau = total_tau).copy()
    s_clamped_phase = evolve_to_equilbrium(s = s, W = W, d = d, beta = beta, eps = eps, total_tau = total_tau).copy()
    dW = weight_update(W, W_exists, s_free_phase, s_clamped_phase)
    dW_list.append(dW.std())
    W += dW
plot(dW_list)