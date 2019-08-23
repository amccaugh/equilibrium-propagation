#%%
# -*- coding: utf-8 -*-

# Functionality verified:
  # Verified that generated W and W_mask are as described by Scellier
  # Verified that 

import torch
import numpy as np
from eqp.model_Scellier import EQP_Network
from eqp.model_Scellier import MNIST_Wrapper

np.set_printoptions(precision=2, linewidth=100)
torch.set_printoptions(precision=2, linewidth=100)

layer_sizes = [784,500,10]
batch_size = 20
learning_rate = [.1,.05]
beta = .5
eps = .5
n_iter = [20, 4]
seed = 0
num_epochs = 25
device='cuda:0'
dtype=torch.float

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)

network = EQP_Network(layer_sizes, batch_size, eps, n_iter, seed, device, dtype)
network.initialize_state()
network.initialize_weight_matrix(kind='Layered',symmetric=True)
network.initialize_biases()
network.initialize_persistant_particles(n_train=50000)

n_train = int(50000/batch_size)
n_test = int(10000/batch_size)
dataset = MNIST_Wrapper(batch_size, device, n_train=50000)

training_error = 0
for i in range(n_train):
    [x, y], _ = dataset.get_training_batch()
    network.set_x_state(x)
    network.evolve_to_equilibrium(y,0)
    training_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
training_error = 1-(float(training_error)/50000)
test_error = 0
for i in range(n_test):
    x, y = dataset.get_test_batch()
    network.set_x_state(x)
    network.evolve_to_equilibrium(y,0)
    test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
test_error = 1-(float(test_error)/10000)
print('Model initialized.\n  Training error: {}\n  Test error: {}'\
      .format(training_error, test_error))

for epoch in range(1,num_epochs+1):
    for i in range(n_train):
        [x, y], index = dataset.get_training_batch()
        network.train_batch(x, y, index, beta, learning_rate)
    training_error = 0
    for i in range(n_train):
        [x, y], index = dataset.get_training_batch()
        network.s = network.persistant_particles[index].clone()
        network.set_x_state(x)
        network.evolve_to_equilibrium(y, 0)
        training_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
    test_error = 0
    for i in range(n_test):
        x, y = dataset.get_test_batch()
        network.set_x_state(x)
        network.evolve_to_equilibrium(y, 0)
        test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
    print('Epoch {} complete.\n  Training error: {}\n  Test error: {}'\
          .format(epoch, 1-(float(training_error)/50000),1-(float(test_error)/10000)))