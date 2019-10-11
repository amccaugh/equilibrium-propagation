#%%
# -*- coding: utf-8 -*-
import torch
import numpy as np
from eqp.model_Scellier import EQP_Network
from eqp.model_Scellier import MNIST_Scellier
from eqp.model_Scellier import Linear
import datetime
import pickle
import time

def printTime(t_0, msg='Time taken: ', n_tabs=0, n_nl=0, offset=True):
    if offset:
        t = time.time()-t_0
    else:
        t = t_0
    unit = None
    tabs = ''.join(['\t']*n_tabs)
    nl = ''.join(['\n']*n_nl)
    if t<1e-3:
        unit='us'
        t *= 1e6
    elif t<1:
        unit='ms'
        t *= 1e3
    elif t<60:
        unit = 's'
    elif t<(60**2):
        unit = 'm'
        t /= 60
    else:
        unit = 'h'
        t /= (60**2)
    print(tabs+msg+('%.01f'%t)+unit+'.'+nl)

layer_sizes = [20,15,15,15,15,15,15,15,15,10]
batch_size = 20
beta = 1
eps = .5
n_iter = [20,4]
num_epochs = 20
device = 'cpu'
dtype = torch.float
max_swconn = 1000
swconn_increment = 10
seed = 0
n_train_ex = 10000
n_test_ex = 2000
Error = {
        'proportion connections': [],
        'initial training error': [],
        'initial test error': [],
        'cumulative training error': [],
        'cumulative test error': [],
        'final training error': [],
        'final test error': [],
        'per-layer cumulative correction': [],
        'clustering coefficient': [],
        'characteristic path length': []}


np.random.seed(seed=seed)
torch.manual_seed(seed=seed)
np.set_printoptions(precision=2, linewidth=100)
torch.set_printoptions(precision=2, linewidth=100)

print('Initializing network for the first time.')
t_0 = time.time()
network = EQP_Network(layer_sizes, batch_size, eps, n_iter, seed=seed, device=device, dtype=dtype)
n_train = int(n_train_ex/batch_size)
n_test = int(n_test_ex/batch_size)
dataset = Linear(batch_size, device, n_train=n_train, n_test=n_test)
network.initialize_weight_matrix(kind='smallworld', symmetric=True, num_swconn=sw_conn)
W_init = network.W.clone()
print('\tDone with initialization.')
printTime(t_0, n_tabs=1, n_nl=1)

print('Calculating untrained test error rate.')
t_0 = time.time()
network.initialize_state()
network.initialize_biases()
network.initialize_persistant_particles(n_particles=n_train+n_test)
test_error = 0
for i in range(n_test):
    [x, y], index = dataset.get_test_batch()
    network.set_x_state(x)
    network.evolve_to_equilibrium(y, 0)
    test_error += torch.norm(network.s-y)
test_error /= n_test_ex
print('\tUntrained test error rate: %.06f'%test_error)
for i in range(int(max_swconn/swconn_increment)):
    t_nc = time.time()
    print('Beginning testing with %d small-world connections.'%i*swconn_increment)
    Error['proportion connections'.append]