#%%
# -*- coding: utf-8 -*-
from eqp.model_Scellier import MNIST_Scellier
dataset = MNIST_Scellier(1,'cpu')


#%%

#%%
# Train network with 3 hidden layers

import torch
import numpy as np
from eqp.model_Scellier import EQP_Network
from eqp.model_Scellier import MNIST_Scellier
import datetime
import pickle
import time

np.set_printoptions(precision=2, linewidth=100)
torch.set_printoptions(precision=2, linewidth=100)

layer_sizes = [784, 500, 500, 500, 10]
batch_size = 20
learning_rate = [.128, .032, .008, .002]
beta = 1
eps = .5
n_iter = [500, 8]
num_epochs = 160
device='cuda:0'
dtype=torch.float

Error = {'seed': [],
         'training error': [],
         'test error': []}

for seed in [1]:
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
    dataset = MNIST_Scellier(batch_size, device, n_train=n_train_ex, n_test=n_test_ex)
    training_error, test_error = 0,0
    for epoch in range(1,num_epochs+1):
        training_error = 0
        for i in range(n_train):
            t_0 = time.time()
            [x, y], index = dataset.get_training_batch()
            training_error += network.train_batch(x, y, index, beta, learning_rate)
            #print('\t%e'%(time.time()-t_0))
        test_error = 0
        for i in range(n_test):
            [x, y], index = dataset.get_test_batch()
            network.use_persistant_particle(index)
            network.set_x_state(x)
            network.evolve_to_equilibrium(y, 0)
            network.set_persistant_particle(index, network.s)
            test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
        print('Epoch {} complete.\n  Training error: {}\n  Test error: {}'\
                  .format(epoch, 1-(float(training_error)/n_train_ex),1-(float(test_error)/n_test_ex)))
        Error['seed'].append(seed)
        Error['training error'].append(1-(float(training_error)/n_train_ex))
        Error['test error'].append(1-(float(test_error)/n_test_ex))
    print('Seed: %d. Final training error: %f. Final test error: %f.'\
          %(seed,1-(float(training_error)/n_train_ex),1-(float(test_error)/n_test_ex)))
    
dt = datetime.datetime.now()
filename = r'MNIST_{}-{}-{}-{}-{}.pickle'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute)
pickle.dump(Error, open(filename,'wb'))




#%%
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

for seed in [1]:
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
    dataset = MNIST_Scellier(batch_size, device, n_train=n_train_ex, n_test=n_test_ex)
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
        network.use_persistant_particle(index)
        network.set_x_state(x)
        network.evolve_to_equilibrium(y, 0)
        network.set_persistant_particle(index, network.s)
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


 #%%
 # Script to look at error rate of network with various seeds

import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

font = {'family':'Keyboard','weight':'normal','size':16}
mpl.rc('font',**font)

filename = r'C:\Users\jig\Documents\GitHub\equilibrium-propagation\mnist_data_scellier\MNIST_2019-8-26-7-0.pickle'

Error = pickle.load(open(filename, 'rb'))
Seeds = Error['seed']
Training_error = Error['training error']
Test_error = Error['test error']

print('Training error stats:')
print('\tMean: %e\n\tStd. Deviation: %e\n\tMax: %e at seed %d\n\tMin: %e at seed %d'
      %(np.mean(Training_error),np.std(Training_error),max(Training_error),\
        Seeds[np.argmax(Training_error,axis=0)],min(Training_error),\
        Seeds[np.argmin(Training_error,axis=0)]))

print('Test error stats:')
print('\tMean: %e\n\tStd. Deviation: %e\n\tMax: %e at seed %d\n\tMin: %e at seed %d'
      %(np.mean(Test_error),np.std(Test_error),max(Test_error),\
        Seeds[np.argmax(Test_error,axis=0)],min(Test_error),\
        Seeds[np.argmin(Test_error,axis=0)]))

_, axl = plt.subplots()
axl.plot(Seeds,Test_error,'r')
axl.set_ylabel('Test error rate')
axl.yaxis.label.set_color('r')
axr = axl.twinx()
axr.plot(Seeds,Training_error,'b')
axr.set_ylabel('Training error rate')
axr.yaxis.label.set_color('b')
plt.title('Error rate vs seed of network with Scellier setup')
plt.xlabel('Seed')












