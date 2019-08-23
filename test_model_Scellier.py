#%%
# -*- coding: utf-8 -*-

# Functionality verified:
  # Verified that generated W and W_mask are as described by Scellier
  # Verified that 

import torch
import numpy as np
from eqp.model_Scellier import EQP_Network
from eqp.model_Scellier import MNIST_Scellier
import datetime
import pickle

np.set_printoptions(precision=2, linewidth=100)
torch.set_printoptions(precision=2, linewidth=100)

layer_sizes = [784,500,10]
batch_size = 20
learning_rate = [.1,.05]
beta = 1
eps = .5
n_iter = [20, 4]
#seed = 0
num_epochs = 25
device='cuda:0'
dtype=torch.float

Error = {'seed': [],
         'training error': [],
         'test error': []}

for seed in range(150):
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)
    
    network = EQP_Network(layer_sizes, batch_size, eps, n_iter, seed, device, dtype)
    network.initialize_state()
    network.initialize_weight_matrix(kind='Layered',symmetric=True)
    network.initialize_biases()
    network.initialize_persistant_particles(n_particles=60000)
    
    n_train_ex = 50000
    n_test_ex = 10000
    n_train = int(n_train_ex/batch_size)
    n_test = int(n_test_ex/batch_size)
    dataset = MNIST_Scellier(batch_size, device)#, n_train=n_train_ex, n_test=n_test_ex)
    """
    training_error = 0
    for i in range(n_train):
        [x, y], _ = dataset.get_training_batch()
        network.set_x_state(x)
        network.evolve_to_equilibrium(y,0)
        training_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
    training_error = 1-(float(training_error)/n_train_ex)
    test_error = 0
    for i in range(n_test):
        [x, y], _ = dataset.get_test_batch()
        network.set_x_state(x)
        network.evolve_to_equilibrium(y,0)
        test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
    test_error = 1-(float(test_error)/n_test_ex)
    print('Model initialized.\n  Training error: {}\n  Test error: {}'\
          .format(training_error, test_error))
    """
    training_error, test_error = 0,0
    for epoch in range(1,num_epochs+1):
        training_error = 0
        for i in range(n_train):
            [x, y], index = dataset.get_training_batch()
            training_error += network.train_batch(x, y, index, beta, learning_rate)
        """
        for i in range(n_train):
            [x, y], index = dataset.get_training_batch()
            network.s = network.persistant_particles[index].clone()
            network.set_x_state(x)
            network.evolve_to_equilibrium(y, 0)
            training_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
        """
    test_error = 0
    for i in range(n_test):
        [x, y], index = dataset.get_test_batch()
        network.s = network.persistant_particles[index].clone()
        network.set_x_state(x)
        network.evolve_to_equilibrium(y, 0)
        network.persistant_particles[index] = network.s.clone()
        test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
        #print('Epoch {} complete.\n  Training error: {}\n  Test error: {}'\
        #      .format(epoch, 1-(float(training_error)/n_train_ex),1-(float(test_error)/n_test_ex)))
    Error['seed'].append(seed)
    Error['training error'].append(1-(float(training_error)/n_train_ex))
    Error['test error'].append(1-(float(test_error)/n_test_ex))
    print('Seed: %d. Final training error: %f. Final test error: %f.'\
          %(seed,1-(float(training_error)/n_train_ex),1-(float(test_error)/n_test_ex)))
    
dt = datetime.datetime.now()
filename = r'MNIST_{}-{}-{}-{}-{}.pickle'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute)
pickle.dump(Error, open(filename,'wb'))