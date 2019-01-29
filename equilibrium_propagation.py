import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import os

#device = torch.device('cpu'); torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda'); torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_dtype(torch.float)
dtype = torch.float

# TODO / things to try:
# - Divide weight matrix by 10
# - Disable state clipping / allow neuron states to be negative (Bengio STDP-compatible allows it!)
# - Try small-world style W_mask (~N/2^k W_mask per layer to layers k distance away)
# Implement randomize_beta (beta gets chosen from randn(1) and see if it helps

#%% Check pytorch can use GPU - https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
#torch.cuda.current_device()
#torch.cuda.device(0)
#torch.cuda.device_count()
#torch.cuda.get_device_name(0)
#torch.cuda.empty_cache()


#%% Neuron implementation
# Following pytorch example from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# Helpful short and to-the-point torch tutorial: https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
torch.manual_seed(seed = 0)

layer_sizes = [28*28, 500, 10]
layer_indices = np.cumsum([0] + layer_sizes)
num_neurons = sum(layer_sizes)

# Set up indices of neurons for easy access later
ix = slice(0, layer_indices[1])
iy = slice(layer_indices[-2], layer_indices[-1])
ih = slice(layer_indices[1], layer_indices[-2])
ihy = slice(layer_indices[1], layer_indices[-1])

# Create masks for state, because updating slices of a matrix is slow
mx = torch.zeros([1, num_neurons])
my = torch.zeros([1, num_neurons])
mh = torch.zeros([1, num_neurons])
mhy = torch.zeros([1, num_neurons])
mx[:,ix] = 1
my[:,iy] = 1
mh[:,ih] = 1
mhy[:,ihy] = 1

def initialize_weight_matrix(layer_sizes, seed = None, kind = 'layered', symmetric = True):
    np.random.seed(seed = seed)
    W = np.zeros([num_neurons,num_neurons])
    W_mask = np.zeros(W.shape, dtype = np.int32)
    if kind == 'layered':
        # Initialize weights matrix, connecting one layer to the next
        for n in range(len(layer_sizes)-1): # Make weights only exist from one layer to the next
            wll = np.random.randn(layer_sizes[n+1],layer_sizes[n]) # The weight matrix for one layer to the next
            wll2 = np.random.randn(layer_sizes[n],layer_sizes[n+1]) # The weight matrix for the reverse
            wll *= np.sqrt(2/(layer_sizes[n]))     # Glorot-Bengio (Xavier) weight normalization
            wll2 *= np.sqrt(2/(layer_sizes[n]))     # Glorot-Bengio (Xavier) weight normalization
        #    wll *= 0.2; wll2 *= 0.2
            i = layer_indices[n+1]
            j = layer_indices[n]
            di = wll.shape[0]
            dj = wll.shape[1]
            W[i:i+di, j:j+dj] = wll
            W[j:j+dj, i:i+di] = wll2
            W_mask[i:i+di, j:j+dj] = wll*0 + 1
#            W_mask[j:j+dj, i:i+di] = wll2*0 + 1
        if symmetric == True:
            W[W_mask.T] *= 0
            W += W.T # Make weights symmetric
        W_mask += W_mask.T
    elif kind == 'fc':
        # Create a fully-connected weight matrix
        W = np.random.randn(num_neurons,num_neurons) # The weight matrix for one layer to the next
        W_mask += 1
        if symmetric == True:
            W = np.tril(W) + np.tril(W, k = -1).T
    elif kind == 'smallworld':
        pass
            
    W = torch.from_numpy(W).float().to(device) # Convert to float Tensor
    W_mask = torch.from_numpy(W_mask).float().to(device) # .byte() is the closest thing to a boolean tensor pytorch has
     # Line up dimensions so that the zeroth dimension is the batch #
    W = W.unsqueeze(0)
    W_mask = W_mask.unsqueeze(0)
    return W, W_mask


def random_initial_state(batch_size = 7):
    s = torch.rand(batch_size, num_neurons)
    return s

def rho(s):
    return torch.clamp(s,0,1)

def rho_old(s):
    return np.clip(s,0,1)

def rhoprime(s):
    rp = torch.zeros(s.shape)
    rp[(0<=s) & (s<=1)] = 1 # SUPER IMPORTANT! if (0<s) & (s<1) zeros + ones cannot escape
    return rp

def E(s, W):
    term1 = 0.5*torch.sum(s*s, dim = 1)
    rho_s = rho(s)
    term2 = torch.matmul(rho_s.unsqueeze(2), rho_s.unsqueeze(1))
    term2 *= W
    term2 = -0.5 * torch.sum(term2, dim = [1,2])
#    term3 = -np.sum([b[i]*rho(s[i]) for i in range(len(b))])
    return term1 + term2 # + term3
    
def C(s, d):
    y = s*my
    return 0.5*torch.norm(y-d, dim = 1)**2


def F(s, W, beta, d):
    if beta == 0:
        return E(s, W)
    return E(s, W) + beta*C(s, d = d) # + term3

def step(s, W, eps, beta, d):
    # s - shape (batch_size, num_neurons)
    # W - shape (num_neurons, num_neurons)
    # beta - constant
    # d - shape (batch_size, num_neurons)
#    %%timeit
#    Rs = (W @ rho(s).unsqueeze(2)).squeeze() # Slow, correct
    Rs = (rho(s) @ W).squeeze() # Fast, but reliant on W being symmetric
    # Rs - shape (batch_size, num_neurons)
    dEds = eps*(Rs - s) # dE/ds term, multiplied by dt (epsilon)
    dEds *= mhy # Mask dEds so it only adds to h and y units
    s += dEds
    if beta != 0:
        dCds = eps*beta*(d - s) # beta*dC/ds weak-clamping term, mul tiplied by dt (epsilon)
        dCds *= my # Mask dCds so it only adds to y (output) units
        s += dCds
    # Clipping prevents states from becoming negative due to bad (Euler) time integration
    # Also, clipping = equivalent to multiplying R(s) by rhoprime(s) when beta = 0
    torch.clamp(s, 0, 1, out = s)
    return s


def evolve_to_equilbrium(s, W, d, beta, eps, total_tau,
                         state_list = None, energy_list = None):
    # If state_recorder is passed an (empty) list, it will append each state to it
    num_steps = int(total_tau/eps)
    for n in range(num_steps):
        step(s, W, eps = eps, beta = beta, d = d)
        if state_list is not None: states.append(s.numpy().copy())
        if energy_list is not None: energies.append(F(s, W, beta, d).numpy().copy())
    return s

    
def plot_states_and_energy(states, energies):
    states_batch = states
    energies_batch = energies
    for n in range(np.array(states_batch).shape[1]):
        states = np.array(states_batch)[:,n,:]
        energies = np.array(energies_batch)[:,n]
        # Plot states
        t = np.linspace(0,len(states) * eps, len(states))
        fig, ax = plt.subplots(2,1, sharex = True)
        ax[0].plot(t, np.array(states)[:,ih],'r')
        ax[0].plot(t, np.array(states)[:,ix],'g')
        ax[0].plot(t, np.array(states)[:,iy],'b')
        ax[0].set_ylabel('State values')
        
        ax[1].plot(t, np.array(energies),'b.-')
        ax[1].set_ylabel('Energy F')
        ax[1].set_xlabel('Time (t/tau)')


# Weight update
def weight_update(W, W_mask, beta, s_free_phase, s_clamped_phase):
    # W_mask = matrix of shape(W) with 1s or zeros based on 
    # whether the connection / weight between i and j exists
    term1 = torch.unsqueeze(rho(s_clamped_phase),dim = 2) @ torch.unsqueeze(rho(s_clamped_phase),dim = 1) # This also works instead of einsum
    term2 = torch.unsqueeze(rho(s_free_phase), dim = 2) @ torch.unsqueeze(rho(s_free_phase), dim = 1) # This also works instead of einsum
    dW = 1/beta*(term1 - term2)
    dW *= W_mask
    return dW


def update_weights(W, beta, s_free_phase, s_clamped_phase, learning_rate = 1):
    dW = weight_update(W, W_mask, beta, s_free_phase, s_clamped_phase)
    W += torch.mean(dW, dim = 0)*learning_rate
    return W


def train_batch(s,W,d, beta, eps, total_tau, learning_rate):
    s = evolve_to_equilbrium(s = s, W = W, d = None, beta = 0, eps = eps, total_tau = total_tau)
    s_free_phase = s.clone()
    
    s = evolve_to_equilbrium(s = s, W = W, d = d, beta = 1, eps = eps, total_tau = total_tau)
    s_clamped_phase = s.clone()
    
    W = update_weights(W, beta, s_free_phase, s_clamped_phase, learning_rate = learning_rate)
    return s, W


def convert_dataset_batch(data, target, batch_size):
    """ Convert the dataset "data" and "target" variables to s and d """
    data, target = data.to(device), target.to(device)
    data = data.reshape([batch_size, 28*28]) # Flatten
    d_target = torch.zeros([batch_size, 10])
    for n in range(batch_size):
        d_target[n, target[n]] = 1 # Convert to one-hot
        
    # Setup intitial state s and target d
    s = random_initial_state(batch_size = batch_size)
    s[:,ix] = data
    d = torch.zeros(s.shape)
    d[:,iy] = d_target
    return s,d


def validate(dataset, num_samples_to_test = 1000):
    """ Returns the % error validated against the training or test dataset """
    batch_size = 1000
    train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)
    num_samples_evaluated = 0
    num_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        s,d = convert_dataset_batch(data,target, batch_size)
        s = evolve_to_equilbrium(s = s, W = W, d = None, beta = 0, eps = eps, total_tau = total_tau)
        compared = s[:,iy].argmax(dim = 1) == d[:,iy].argmax(dim = 1)
        num_samples_evaluated += batch_size
        num_correct += torch.sum(compared)
        if num_samples_evaluated > num_samples_to_test:
            break
    return (1-num_correct.item()/num_samples_evaluated)*100

#%% Run algorithm

#seed = 2
#eps = 0.01
#batch_size = 20
#beta = 0.1
#total_tau = 10
#learning_rate = 1e-2
#W, W_mask = initialize_weight_matrix(layer_sizes, seed = seed)
#T = target_matrix(seed = seed)
#
#states = []
#energies = []
#costs = []
#for n in range(100):
#    s = random_initial_state(batch_size = batch_size)
#    d = generate_targets(s, T)
#    
#    s = evolve_to_equilbrium(s = s, W = W, d = None, beta = 0, eps = eps, total_tau = total_tau)
##                             state_list = states, energy_list = energies)
#    s_free_phase = s.clone()
#    
#    s = evolve_to_equilbrium(s = s, W = W, d = d, beta = 1, eps = eps, total_tau = total_tau)
##                             state_list = states, energy_list = energies)
#    s_clamped_phase = s.clone()
##    plot_states_and_energy(states, energies)
#    
#    W = update_weights(W, beta, s_free_phase, s_clamped_phase, learning_rate = learning_rate)
#    costs.append(torch.mean(C(s, d)).item())

#plot(costs)

#%% Plot energies
#seed = 1
#eps = 0.01
#batch_size = 1
#beta = 0.1
#total_tau = 10
#learning_rate = 1e-3
#W, W_mask = initialize_weight_matrix(layer_sizes, seed = seed)
#T = target_matrix(seed = seed)
#
#states = []
#energies = []
#costs = []
#s = random_initial_state(batch_size = batch_size)
##    x = s[:,ix]
##    y = s[:,iy]
#d = generate_targets(s, T)
#
#s = evolve_to_equilbrium(s = s, W = W, d = None, beta = 0, eps = eps, total_tau = total_tau,
#                     state_list = states, energy_list = energies)
#s_free_phase = s.clone()
#
#s = evolve_to_equilbrium(s = s, W = W, d = d, beta = 1, eps = eps, total_tau = total_tau,
#                     state_list = states, energy_list = energies)
#s_clamped_phase = s.clone()
#plot_states_and_energy(states, energies)
#
#W = update_weights(W, beta, s_free_phase, s_clamped_phase, learning_rate = 1e-3)
#costs.append(torch.mean(C(s, d)).item())
    
#%% Thing

    
seed = 2
eps = 0.5
batch_size = 20
beta = 0.1
total_tau = 10
learning_rate = 0.01
W, W_mask = initialize_weight_matrix(layer_sizes, seed = seed)
#T = target_matrix(seed = seed)

# Setup MNIST data loader
data_path = os.path.realpath('./mnist_data/')
train_dataset = datasets.MNIST(data_path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
#                       transforms.Normalize((0.5,), (0.3081,))
                   ]))
test_dataset = datasets.MNIST(data_path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
#                       transforms.Normalize((0.5,), (0.3081,))
                   ]))
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

# Run training loop
costs = []
error = []
for epoch in tqdm(range(20)):
    for batch_idx, (data, target) in enumerate(train_loader):
#        epoch = 1
        s,d = convert_dataset_batch(data,target, batch_size)
        s,W = train_batch(s, W, d, beta, eps, total_tau, learning_rate)
#        cost = torch.mean(C(s, d))
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cost.item()))
        if batch_idx % 500 == 0:
            train_error = validate(train_dataset, num_samples_to_test = 10000)
            test_error = validate(test_dataset, num_samples_to_test = 10000)
            print('Validation:  Training error %0.1f%% / Test error %0.1f%%' % (train_error, test_error))
            error.append([batch_idx, train_error, test_error])
#        costs.append(cost)
