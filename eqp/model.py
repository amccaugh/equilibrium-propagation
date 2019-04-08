# TO Fself.ix: In non-layered connections, the neurons are likely saturating due to
# large amounts of integrated input from all the other neurons; check energy
# diagrams

# TODO: Convert to class so self.mhy, self.mx, etc are available as local variables

import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import os
from tqdm import tqdm

#device = torch.device('cpu'); torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda'); torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_dtype(torch.float)
dtype = torch.float


# TODO / things to try:
# Divide weight matrix by 10
# Implement randomize_beta (beta gets chosen from randn(1) and see if it helps

# Check pytorch can use GPU - https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
#torch.cuda.current_device()
#torch.cuda.device(0)
#torch.cuda.device_count()
#torch.cuda.get_device_name(0)
#torch.cuda.empty_cache()


#%% Neuron implementation
# Following pytorch example from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# Helpful short and to-the-point torch tutorial: https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/


def rho(s):
    return torch.clamp(s,0,1)

def rhoprime(s):
    rp = torch.zeros(s.shape)
    rp[(0<=s) & (s<=1)] = 1 # SUPER IMPORTANT! if (0<s) & (s<1) zeros + ones cannot escape
    return rp


class EP_Network(object):
    
    def __init__(self, eps = 0.5, total_tau = 10, batch_size = 20, seed = None, layer_sizes = [28*28, 500, 10]):
        if seed is not None:
            torch.manual_seed(seed = seed)
            np.random.seed(seed = seed)

        self.layer_sizes = layer_sizes
        self.layer_indices = np.cumsum([0] + self.layer_sizes)
        self.num_neurons = sum(self.layer_sizes)
        self.batch_size = batch_size
        self.eps = eps
        self.total_tau = total_tau

        # Set up indices of neurons for easy access later
        self.ix = slice(0, self.layer_indices[1])
        self.iy = slice(self.layer_indices[-2], self.layer_indices[-1])
        self.ih = slice(self.layer_indices[1], self.layer_indices[-2])
        self.ihy = slice(self.layer_indices[1], self.layer_indices[-1])

        # Create masks for state, because updating slices of a matrix is slow
        self.mx = torch.zeros([1, self.num_neurons])
        self.my = torch.zeros([1, self.num_neurons])
        self.mh = torch.zeros([1, self.num_neurons])
        self.mhy = torch.zeros([1, self.num_neurons])
        self.mx[:,self.ix] = 1
        self.my[:,self.iy] = 1
        self.mh[:,self.ih] = 1
        self.mhy[:,self.ihy] = 1

    def initialize_weight_matrix(self, layer_sizes, seed = None, kind = 'layered', symmetric = True,
                                 density = 0.5, # Value from 0 to 1, used for 'smallworld' and 'sparse' connectivity
                                 ):
        W = np.zeros([self.num_neurons,self.num_neurons], dtype = np.float)
        W_mask = np.zeros(W.shape, dtype = np.int32)
        if kind == 'layered':
            # Initialize weights matrix, connecting one layer to the next
            for n in range(len(self.layer_sizes)-1): # Make weights only exist from one layer to the next
                wll = np.random.randn(self.layer_sizes[n+1],self.layer_sizes[n]) # The weight matrix for one layer to the next
                wll2 = np.random.randn(self.layer_sizes[n],self.layer_sizes[n+1]) # The weight matrix for the reverse
                wll *= np.sqrt(2/(self.layer_sizes[n]))     # Glorot-Bengio (Xavier) weight normalization
                wll2 *= np.sqrt(2/(self.layer_sizes[n]))     # Glorot-Bengio (Xavier) weight normalization
            #    wll *= 0.2; wll2 *= 0.2
                i = self.layer_indices[n+1]
                j = self.layer_indices[n]
                di = wll.shape[0]
                dj = wll.shape[1]
                W[i:i+di, j:j+dj] = wll
                W[j:j+dj, i:i+di] = wll2
                W_mask[i:i+di, j:j+dj] = wll*0 + 1
                W_mask[j:j+dj, i:i+di] = wll2*0 + 1
        elif kind == 'fc':
            # Create a fully-connected weight matrix
            W = np.random.randn(self.num_neurons,self.num_neurons) # The weight matrix for one layer to the next
            W *= np.sqrt(2/(self.num_neurons))
            W_mask += 1
        elif kind == 'smallworld':
            # Create a small-world-connectivity matrix
            pass
        elif kind == 'sparse':
            # Creates random connections.  If symmetric=False, most connections will
            # be simplex, not duplex.  Uses density parameter
            W_mask = np.random.binomial(n = 1, p = density, size = W.shape)
            W = np.random.randn(self.num_neurons,self.num_neurons)*W_mask # The weight matrix for one layer to the next
        
        if symmetric == True:
            W = np.tril(W) + np.tril(W, k = -1).T
        # Make sure trace elements are zero so neurons don't self-reference
        W -= np.eye(self.num_neurons, dtype = np.float)*W
        W_mask -= np.eye(self.num_neurons, dtype = np.int32)*W_mask
        # Disconnect input and output neurons
        W_mask[self.iy,self.ix] *= 0
        W_mask[self.ix,self.iy] *= 0
        W *= W_mask
        # Convert to Tensor format on the correct device (CPU/GPU)
        W = torch.from_numpy(W).float().to(device) # Convert to float Tensor
        W_mask = torch.from_numpy(W_mask).float().to(device) # .byte() is the closest thing to a boolean tensor pytorch has
         # Line up dimensions so that the zeroth dimension is the batch #
        self.W = W.unsqueeze(0)
        self.W_mask = W_mask.unsqueeze(0)
        return self.W, self.W_mask


    def randomize_initial_state(self, batch_size):
        self.s = torch.rand(batch_size, self.num_neurons)
        return self.s

    def set_x_state(self, x):
                # # Setup intitial state s and target d
        # # s = self.randomize_initial_state()
        self.s[:,self.ix] = x
        # d = torch.zeros(s.shape)
        # d[:,self.iy] = d_target
        # self.s = s
        # return s,d

    def set_y_state(self, y):
        self.s[:,self.iy] = y

    def E(self):
        term1 = 0.5*torch.sum(self.s*self.s, dim = 1)
        rho_s = rho(self.s)
        term2 = torch.matmul(rho_s.unsqueeze(2), rho_s.unsqueeze(1))
        term2 *= self.W
        term2 = -0.5 * torch.sum(term2, dim = [1,2])
    #    term3 = -np.sum([b[i]*rho(s[i]) for i in range(len(b))])
        return term1 + term2 # + term3
        
    def C(self, y_target):
        # y = self.s*self.my
        y = self.s[:,self.iy]
        return 0.5*torch.norm(y-y_target, dim = 1)**2


    def F(self, beta, y_target):
        if beta == 0:
            return self.E()
        return self.E() + beta*self.C(y_target = y_target) # + term3


    def step(self, beta, y_target):
        # s - shape (batch_size, self.num_neurons)
        # W - shape (self.num_neurons, self.num_neurons)
        # beta - constant
        # d - shape (batch_size, self.num_neurons)
    #    %%timeit
    #    Rs = (W @ rho(s).unsqueeze(2)).squeeze() # Slow, correct
        # s = self.s
        # W = self.W
        Rs = (rho(self.s) @ W).squeeze() # Fast, but reliant on W being symmetric
        # Rs - shape (batch_size, self.num_neurons)
        dEds = self.eps*(Rs - self.s) # dE/ds term, multiplied by dt (epsilon)
        dEds *= self.mhy # Mask dEds so it only adds to h and y units
        self.s += dEds
        if beta != 0:
            # dCds = self.eps*beta*(d - self.s) # beta*dC/ds weak-clamping term, mul tiplied by dt (epsilon)
            # dCds *= self.my # Mask dCds so it only adds to y (output) units
            # self.s += dCds
            dCds = self.eps*beta*(y_target - self.s[:,self.iy])
            self.s[:,self.iy] += dCds
        # Clipping prevents states from becoming negative due to bad (Euler) time integration
        # Also, clipping = equivalent to multiplying R(s) by rhoprime(s) when beta = 0
        torch.clamp(self.s, 0, 1, out = self.s)
        return self.s


    def evolve_to_equilbrium(self, y_target, beta,
                             state_list = None, energy_list = None):
        # If state_recorder is passed an (empty) list, it will append each state to it
        num_steps = int(self.total_tau/self.eps)
        for n in range(num_steps):
            self.step(beta = beta, y_target = y_target)
            if state_list is not None: state_list.append(self.s.cpu().numpy().copy())
            if energy_list is not None: energy_list.append(self.F(beta, y_target).cpu().numpy().copy())
        return self.s

        
    def plot_states_and_energy(self, state_list, energy_list):
        state_list_batch = state_list
        energy_list_batch = energy_list
        for n in range(np.array(state_list_batch).shape[1]):
            state_list = np.array(state_list_batch)[:,n,:]
            energy_list = np.array(energy_list_batch)[:,n]
            # Plot state_list
            t = np.linspace(0,len(state_list) * eps, len(state_list))
            fig, ax = plt.subplots(2,1, sharex = True)
            lines1 = ax[0].plot(t, np.array(state_list)[:,self.ih],'b--', label = 'h_hidden')
            lines2 = ax[0].plot(t, np.array(state_list)[:,self.ix],'g', label = 'x_input')
            lines3 = ax[0].plot(t, np.array(state_list)[:,self.iy],'r', label = 'y_output')
            ax[0].set_ylabel('State values')
            ax[0].legend([lines1[0], lines2[0], lines3[0]], ['h_hidden','x_input','y_output'])
            
            ax[1].plot(t, np.array(energy_list),'b.-')
            ax[1].set_ylabel('Energy F')
            ax[1].set_xlabel('Time (t/tau)')


    # Weight update
    def _calculate_weight_update(self, beta, s_free_phase, s_clamped_phase):
        # W_mask = matrix of shape(W) with 1s or zeros based on 
        # whether the connection / weight between i and j exists
        term1 = torch.unsqueeze(rho(s_clamped_phase),dim = 2) @ torch.unsqueeze(rho(s_clamped_phase),dim = 1) # This also works instead of einsum
        term2 = torch.unsqueeze(rho(s_free_phase), dim = 2) @ torch.unsqueeze(rho(s_free_phase), dim = 1) # This also works instead of einsum
        dW = 1/beta*(term1 - term2)
        dW *= self.W_mask
        return dW


    def train_batch(self, x_data, y_target, beta, learning_rate):
        # Perform free phase evolution
        self.set_x_state(x_data)
        self.evolve_to_equilbrium(y_target = None, beta = 0)
        s_free_phase = self.s.clone()
        
        # Perform weakly-clamped phase evolution
        self.set_x_state(x_data)
        self.evolve_to_equilbrium(y_target = y_target, beta = 1)
        s_clamped_phase = self.s.clone()
        
        # Update weights
        dW = self._calculate_weight_update(beta, s_free_phase, s_clamped_phase)
        self.W += torch.mean(dW, dim = 0)*learning_rate
        # return s,W


class Target_MNIST(object):
    def __init__(self):
        # Setup MNIST data loader
        data_path = os.path.realpath('./mnist_data/')
        self.test_dataset = datasets.MNIST(data_path, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
        #                       transforms.Normalize((0.5,), (0.3081,))
                           ]))
        self.train_dataset = datasets.MNIST(data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
        #                       transforms.Normalize((0.5,), (0.3081,))
                           ]))

    def generate_inputs_and_targets(self, batch_size, train = True):
        """ Returns input data x of size x, and an output target state """
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)
        batch_idx, (data, target) = next(enumerate(loader))

        """ Convert the dataset "data" and "target" variables to s and d """
        data, target = data.to(device), target.to(device)
        x_data = data.reshape([batch_size, 28*28]) # Flatten
        y_target = torch.zeros([batch_size, 10])
        for n in range(batch_size):
            y_target[n, target[n]] = 1 # Convert to one-hot
        
        return x_data, y_target





class MNISTDataset(Dataset):
    def __init__(self, train = True):
        data_path = os.path.realpath('./mnist_data/')
        self.dataset = datasets.MNIST(data_path, train=train, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
#                       transforms.Normalize((0.5,), (0.3081,))
                   ]))
        
    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        x_data = img.reshape([28*28])
        y_target = torch.zeros([10])
        y_target[label] = 1 # Convert to one-hot
        return (x_data, y_target)

    def __len__(self):
        return self.dataset.__len__() # of how many examples(images?) you have

    def validate(self, epn, num_samples_to_test = 1000):
        """ Returns the % error validated against the training or test dataset """
        batch_size = epn.batch_size
        dataloader = torch.utils.data.DataLoader(dataset = self, batch_size = batch_size, shuffle=True)
        num_samples_evaluated = 0
        num_correct = 0
        for batch_idx, (x_data, y_target) in enumerate(dataloader):
            epn.randomize_initial_state(batch_size = batch_size)
            epn.set_x_state(x_data)
            s = epn.evolve_to_equilbrium(y_target = None, beta = 0)
            compared = s[:,epn.iy].argmax(dim = 1) == y_target[:].argmax(dim = 1)
            num_samples_evaluated += batch_size
            num_correct += torch.sum(compared)
            if num_samples_evaluated > num_samples_to_test:
                break
        error = (1-num_correct.item()/num_samples_evaluated)
        return error


class LinearMatrixDataset(Dataset):
    def __init__(self, input_size, output_size, length = 10000):
        self.input_size = input_size
        self.output_size = output_size
        self.T = torch.rand((output_size, input_size), dtype = dtype)/5
        self.length = int(length)
        
    def __getitem__(self, index):
        x_data = torch.rand((self.input_size, 1), dtype = dtype).squeeze()/5
        y_target = torch.matmul(self.T,x_data).squeeze()
        return (x_data, y_target)

    def __len__(self):
        return self.length # of how many examples(images?) you have

    def validate(self, epn, num_samples_to_test = 1000):
        batch_size = epn.batch_size
        dataloader = torch.utils.data.DataLoader(dataset = self, batch_size = batch_size, shuffle=True)
        num_samples_evaluated = 0
        total_error = 0
        for batch_idx, (x_data, y_target) in enumerate(dataloader):
            epn.randomize_initial_state(batch_size = batch_size)
            epn.set_x_state(x_data)
            s = epn.evolve_to_equilbrium(y_target = None, beta = 0)
            y = s[:,epn.iy]
            total_error += torch.sum(torch.abs(y-y_target)).item()
#            compared = s[:,epn.iy].argmax(dim = 1) == y_target[:].argmax(dim = 1)
            num_samples_evaluated += batch_size
#            num_correct += torch.sum(compared)
            if num_samples_evaluated > num_samples_to_test:
                break
        error = total_error/num_samples_evaluated/y_target.size()[1]
        return error



    