# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import os
import gzip
import pickle

def rho(s):
    # Neuron activation function.
    # "For our experiments, we choose the hard sigmoid activation function 
    # ρ(si) = 0 ∨ si ∧ 1, where ∨ denotes the max and ∧ the min."
    return torch.clamp(s,0,1)
def rhoprime(s, device):
    # First derivative w.r.t. s of neuron activation function.
    rp = torch.zeros(s.shape).to(device)
    rp[(0<=s) & (s<=1)] = 1
    return rp

class EQP_Network:
    def __init__(self, layer_sizes, batch_size,  eps, n_iter,
                 seed=None, device='cpu', dtype=torch.float):
        
        if seed is not None:
            torch.manual_seed(seed=seed)
            np.random.seed(seed=seed)
        
        self.seed = seed
        self.layer_sizes = layer_sizes
        self.layer_indices = np.cumsum([0] + self.layer_sizes)
        self.num_neurons = sum(self.layer_sizes)
        self.batch_size = batch_size
        self.eps = eps
        self.n_iter = n_iter
        
        self.ix = slice(0,self.layer_indices[1])
        self.iy = slice(self.layer_indices[-2],self.layer_indices[-1])
        self.ihy = slice(self.layer_indices[1],self.layer_indices[-1])
        
        self.device = device
        self.dtype = dtype
        
    def initialize_state(self):
        # initialize state matrix
        self.s = torch.zeros(self.batch_size, self.num_neurons).to(self.device)
        
    def initialize_persistant_particles(self, n_particles=60000):
        self.persistant_particles = []
        for i in range(int(n_particles/self.batch_size)):
            self.persistant_particles.append(torch.zeros(self.s[:,self.ihy].shape).to(self.device))
            
    def use_persistant_particle(self, index):
        self.s[:,self.ihy] = self.persistant_particles[index].clone()
            
    def set_persistant_particle(self, index, state):
        self.persistant_particles[index] = state[:,self.ihy].clone()
        
    def initialize_weight_matrix(self, kind='Layered', symmetric=True):
        # initialize weight and mask to represent interlayer connection strengths
        W = np.zeros((self.num_neurons,self.num_neurons),dtype=np.float32)
        W_mask = np.zeros((self.num_neurons,self.num_neurons),dtype=np.float32)
        # initialize masks to retrieve connection between two layers
        interlayer_connections = []
        for i,j,k in zip(self.layer_indices[:-2],self.layer_indices[1:-1],self.layer_indices[2:]):
            conn = np.zeros(W.shape,dtype=np.float32)
            conn[i:j,j:k] = 1
            conn[j:k,i:j] = 1
            interlayer_connections.append(conn)
        if kind=='Layered':
            # Add connections between every two neurons in adjacent layers
            # No intralayer connections
            for conn in interlayer_connections:
                W_mask += conn
        # Glorot-Bengoi weight initialization, as in Scellier code
        for conn, n_in, n_out in zip(interlayer_connections, self.layer_sizes[:-1], self.layer_sizes[1:]):
            rng = np.random.RandomState(seed=self.seed)
            W += conn*np.asarray(
                    rng.uniform(
                            low=-np.sqrt(6. / (n_in+n_out)),
                            high=np.sqrt(6. / (n_in+n_out)),
                            size=W.shape))
        if symmetric==True:
        # Make W and W_mask symmetrical with zeros on diagonal
            W = np.tril(W,k=-1)+np.tril(W,k=-1).T
            W_mask = np.tril(W_mask,k=-1)+np.tril(W_mask,k=-1).T
        # Convert numpy tensors to pytorch tensors
        self.W = torch.from_numpy(W).float().to(self.device).unsqueeze(0)
        self.W_mask = torch.from_numpy(W_mask).float().to(self.device).unsqueeze(0)
        self.interlayer_connections = [torch.from_numpy(conn).float().to(self.device).unsqueeze(0)\
                                       for conn in interlayer_connections]

    def initialize_biases(self):
        # initialize bias matrix
        self.B = torch.zeros(self.s.shape).to(self.device)

    def set_x_state(self, x):
        # set input neurons of state matrix to x
        self.s[:,self.ix] = x
        
    def set_y_state(self, y):
        # set output neurons of state matrix to y 
        self.s[:,self.iy] = y

    def E(self):
        # Compute energy of current state.
        term1 = .5*torch.sum(self.s*self.s,dim=1)
        rho_s = rho(self.s)
        term2 = torch.matmul(rho_s.unsqueeze(2),rho_s.unsqueeze(1))
        term2 = term2 * self.W
        term2 = -.5*torch.sum(term2,dim=[1,2])
        term3 = -1*torch.sum(self.B*rho_s,dim=1)
        return term1 + term2 + term3
    
    def C(self, y_target):
        # Sum of squared errors
        y = self.s[:,self.iy]
        return .5*torch.norm(y-y_target,dim=1)**2
    
    def F(self, beta, y_target):
        # Total error
        if beta==0:
            return self.E()
        return self.E() + beta*self.C(y_target)
    
    def step(self, beta, y_target):
        Rs = (rho(self.s)@self.W).squeeze() + self.B
        dEds = self.eps*(rhoprime(self.s, self.device)*Rs-self.s)
        dEds[:,self.ix] = torch.zeros(dEds[:,self.ix].shape)
        self.s = self.s + dEds
        if beta != 0:
            dCds = self.eps*beta*(y_target-self.s[:,self.iy])
            self.s[:,self.iy] = self.s[:,self.iy] + dCds
        self.s = torch.clamp(self.s, 0, 1)

    def evolve_to_equilibrium(self, y_target, beta):
        for i in range(self.n_iter[1 if beta else 0]):
            self.step(beta=beta,y_target=y_target)
    
    def calculate_weight_update(self, beta, s_free_phase, s_clamped_phase):
        term1 = torch.unsqueeze(rho(s_clamped_phase),dim=2)@torch.unsqueeze(rho(s_clamped_phase),dim=1)
        term2 = torch.unsqueeze(rho(s_free_phase),dim=2)@torch.unsqueeze(rho(s_free_phase),dim=1)
        dW = (1/beta) * (term1 - term2)
        dW *= self.W_mask
        return dW

    def calculate_bias_update(self, beta, s_free_phase, s_clamped_phase):
        dB = (1/beta) * (rho(s_clamped_phase) - rho(s_free_phase))
        return -dB

    def train_batch(self, x, y, index, beta, learning_rate):
        # initialize state to previously-computed state for this batch
        self.use_persistant_particle(index)
        self.set_x_state(x)
        self.W.requires_grad = True
        self.B.requires_grad = True
        self.evolve_to_equilibrium(None,0)
        E_free = torch.mean(self.E().clone())
        training_error = torch.eq(torch.argmax(self.s[:,self.iy],dim=1),torch.argmax(y,dim=1)).sum()
        self.set_persistant_particle(index, self.s)
        # save state to initialize next time this batch is encountered
        if np.random.randint(0,2): 
            # randomize sign of beta
            beta = beta*-1
        self.evolve_to_equilibrium(y,beta)
        E_clamped = torch.mean(self.E().clone())            
        dW, dB = torch.autograd.grad((E_clamped-E_free), (self.W,self.B), retain_graph=True)
        self.B.requires_grad = False
        self.W.requires_grad = False
        
        dW = dW/beta
        dB = dB/beta
        
        # implement per-layer learning rates
        for lr, conn in zip(learning_rate, self.interlayer_connections):
            dW[conn!=0] = dW[conn!=0]*lr
        self.W = self.W + dW
        
        dB[:,self.ix] = 0
        for lr, i, j in zip(learning_rate, self.layer_indices[1:-1], self.layer_indices[2:]):
            dB[i:j] = dB[i:j]*lr
        self.B = self.B + dB
        return training_error
        
"""
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
        #Returns input data x of size x, and an output target state 
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)
        batch_idx, (data, target) = next(enumerate(loader))

        #Convert the dataset "data" and "target" variables to s and d
        x_data = data.reshape([batch_size, 28*28]) # Flatten
        y_target = torch.zeros([batch_size, 10])
        for n in range(batch_size):
            y_target[n, target[n]] = 1 # Convert to one-hot
        
        return x_data, y_target

class MNIST_Wrapper:
    def __init__(self, batch_size, device, n_train=60000, n_test=10000):
        self.n_train = n_train
        self.n_test = n_test
        self.batch_size = batch_size
        self.training_batches = []
        self.test_batches = []
        self.training_index = 0
        self.test_index = 0
        dataset = Target_MNIST()
        for example in range(int(self.n_train/self.batch_size)):
            x, y = dataset.generate_inputs_and_targets(self.batch_size,True)
            self.training_batches.append([[x.to(device),y.to(device)], example])
        for example in range(int(self.n_test/self.batch_size)):
            x, y = dataset.generate_inputs_and_targets(self.batch_size,False)
            self.test_batches.append([x.to(device),y.to(device)])
    def get_training_batch(self):
        rv = self.training_batches[self.training_index]
        self.training_index = (self.training_index+1)%int(self.n_train/self.batch_size)
        #if self.training_index==0:
        #    np.random.shuffle(self.training_batches)
        return rv
    def get_test_batch(self):
        rv = self.test_batches[self.test_index]
        self.test_index = (self.test_index+1)%int(self.n_test/self.batch_size)
        return rv
"""

class MNIST_Scellier:
    def __init__(self, batch_size, device, n_train=60000, n_test=10000):
        path = r'/home/qittlab/Desktop/jimmy/equilibrium-propagation/mnist.pkl.gz'
            
        f = gzip.open(path, 'rb')
        (x_train,y_train), (x_validate,y_validate), (x_test, y_test) = pickle.load(f, encoding='latin1')
        f.close()
        
        x = list(x_train)+list(x_validate)+list(x_test)
        y = list(y_train)+list(y_validate)+list(y_test)
        for i, yy in zip(range(len(y)),y):
            v = np.zeros((1,10))
            v[0][yy] = 1
            y[i] = v
        x = [torch.from_numpy(xx).squeeze().to(device) for xx in x]
        y = [torch.from_numpy(yy).squeeze().to(device) for yy in y]
        
        self.n_batch_train = int(n_train/batch_size)
        self.n_batch_test = int(n_test/batch_size)
        self.training_batches = []
        self.test_batches = []
        self.training_index = 0
        self.test_index = 0
        
        for batch in range(self.n_batch_train):
            self.training_batches.append(
                     [[torch.stack(x[batch_size*batch:batch_size*(batch+1)],dim=0).float(),
                      torch.stack(y[batch_size*batch:batch_size*(batch+1)],dim=0).float()],
                      batch])
        for batch in range(self.n_batch_train,self.n_batch_train+self.n_batch_test):
            self.test_batches.append(
                    [[torch.stack(x[batch_size*batch:batch_size*(batch+1)],dim=0).float(),
                      torch.stack(y[batch_size*batch:batch_size*(batch+1)],dim=0).float()],
                      batch])
        
    def get_training_batch(self):
        rv = self.training_batches[self.training_index]
        self.training_index = (self.training_index+1)%(self.n_batch_train)
        return rv
    def get_test_batch(self):
        rv = self.test_batches[self.test_index]
        self.test_index = (self.test_index+1)%(self.n_batch_test)
        return rv
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        