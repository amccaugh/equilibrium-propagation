#%%
# -*- coding: utf-8 -*-
from eqp.model_Scellier import MNIST_Scellier
dataset = MNIST_Scellier(1,'cpu')


#%%

#%%
# Train network with 3 hidden layers

def getPotentialConn(network_arg, layer_sizes):
    try:
        network = network_arg.copy()
    except:
        network = network_arg.clone().squeeze()
        network = network.cpu()
    n_vertices = network.shape[0]
    indices = np.cumsum([0]+layer_sizes)
    for i in range(1, len(indices)):
        network[indices[i-1]:indices[i], indices[i-1]:indices[i]] = 1
    potential_conn = .5*(n_vertices**2-np.count_nonzero(network))
    for i in range(1, len(indices)):
        network[indices[i-1]:indices[i], indices[i-1]:indices[i]] = 0
    return int(potential_conn)



import torch
import numpy as np
from eqp.model_Scellier import EQP_Network
from eqp.model_Scellier import MNIST_Scellier
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

np.set_printoptions(precision=2, linewidth=100)
torch.set_printoptions(precision=2, linewidth=100)
 # allow Torch and Numpy to print larger arrays on one line, for ease of viewing

layer_sizes = [784, 500, 500, 500, 10]
 # number of neurons per layer (input, hidden, output)
  # Scellier test 1: [784,500,10]
  # Scellier test 2: [784,500,500,10]
  # Scellier test 3: [784,500,500,500,10]
batch_size = 20
 # how many datapoints to consider during each step of gradient descent
  # 20 in all of Scellier's tests
beta = 1.015
 # clamping factor for weakly-clamped phase
   # 1.0 in all of Scellier's tests
eps = .5
 # size of steps in differential equation
  # 0.5 in all of Scellier's tests
n_iter = [500, 8]
 # number of timesteps in differential equation in free and weakly-clamped phases, respectively
  # Scellier test 1: [20,4]
  # Scellier test 2: [100,6]
  # Scellier test 3: [500,8]
num_epochs = 200
 # number of times to train over full dataset
  # Scellier test 1: 25
  # Scellier test 2: 60
  # Scellier test 3: 160
device='cuda:0'
 # 'cuda:0' for GPU or 'cpu' for CPU
dtype=torch.float
 # datatype for elements of network
sw_conn = 210368
 # number of small-world connections that will be present in network
seed = 0
 # to be passed to Numpy and Torch; network will behave the same when re-run with same seed
n_train_ex = 50000
n_test_ex = 10000
 # number of datapoints to consider
  # 50k training examples and 10k testing examples in Scellier's code
learning_rate = .1

Error = {'beta': [],
         'training error': [],
         'test error': [],
         'layer rates': []}

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)

print('Initializing network for first time.')
t_0 = time.time()
network = EQP_Network(layer_sizes, batch_size, eps, n_iter, seed, device, dtype)
 # store hyperparameters in network
n_train = int(n_train_ex/batch_size)
n_test = int(n_test_ex/batch_size)
 # number of batches for which to train/test
dataset = MNIST_Scellier(batch_size, device, n_train=n_train_ex, n_test=n_test_ex)
 # initialize MNIST dataset
network.initialize_weight_matrix(kind='smallworld', symmetric=True, num_swconn=sw_conn)
 # initialize weight matrices in small-world configuration
W_init = network.W.clone()
 # store weight matrix since Numpy random functions take a long time to run
learning_rates = np.linspace(.01,.26,15)
betas = [1.015]#np.linspace(.5,1.5,100)
W_factor = np.linspace(0,10,100)
 # rates for which to test network performance
print('\tDone with initialization.')
printTime(t_0, n_tabs=1, n_nl=1)

print('Calculating untrained error rate.')
network.initialize_state()
network.initialize_biases()
network.initialize_persistant_particles(n_particles=n_train_ex+n_test_ex)
test_error = 0
for i in range(n_test):
    [x, y], index = dataset.get_test_batch()
    network.use_persistant_particle(index)
    network.set_x_state(x)
    network.evolve_to_equilibrium(y, 0)
    network.set_persistant_particle(index, network.s)
    test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
print(('\tUntrained error rate: %.06f'%(100*(1-(float(test_error)/n_test_ex))))+'%.')

# Currently: testing performance with beta=1.015.

# Next: sweep weight factor from 0 to 10

##Changes made:
   # Increased unclamped n_iter from 500 to 1000 and clamped n_iter from 8 to 100.
     # Hypothesis: due to more neurons in network, the original number of iterations
     #  was insufficient to produce a good approximation of the differential equation.
     # Result: Appeared to stagnate after first epoch in same way that pre-change
     #  networks stagnated after 5 epochs.
     # Further trial: decreasing number of iterations from 500/8 to 100/8.
   # Swept through various values of beta. Around 1.005 appears to work the best.
     # Model is very sensitive to changes in beta; e.g., 1.0455 acheives error of
     # 7.976% but 1.1566 has error of 35%.
   # beta=1.015 appears to be a good choice.
     # Weird phenomenon: graph of error rate vs. beta looks almost like upward-facing
     # parabola, but error rate jumps up around where minimum would be. Likely result of
     # effect where error reduces early on but jumps back up after 5 or so epochs.
     # Note: this is a little bit higher than the beta corresponding to the lowest
     # error rate. This is because for beta less than around 1.005, the error rate
     # appeared to fluctuate widely, whereas above about 1.005 it appeared to increase
     # in a smooth, exponential-like pattern. 1.015 seems therefore less-likely
     # to stop training after some number of epochs.
   # Verified that dataset is being properly reset after each epoch.
   # Verified that persistant particles are being used correctly.
     
## Changes to try:
   # Change training error rate calculation to match test error rate calculation,
   #  as the differences between the two make it hard to tell if error rates are
   #  reasonable.
   # Compute clustering coefficient vs. number of added connections for a
   #  Scellier-style network, as minimum value may not coincide with that of
   #  network for linear dataset.
   # Sweep learning rate around .1 to find the best one.
   # Observe training magnitude to each layer to see if it makes sense.
   # DONE: Make sure dataset is being fully reset after each epoch.
   # DONE: Make sure persistant particles are being used correctly.
   # Try changing beta and epsilon to see what effect it has.

# to do: 24 networks, 50 epochs, around learning rate of .1: np.linspace(.01,.25,24)
#for lr in learning_rates:
for b in betas:
    t_lr = time.time()
    print('Beginning testing with beta of %.04f.'%b)
    Error['beta'].append(b)
    print('\tResetting network:')
    t_0 = time.time()
    network.initialize_state()
    network.initialize_biases()
    network.initialize_persistant_particles(n_particles=n_train_ex+n_test_ex)
    network.W = W_init.clone()
    print('\t\tDone resetting network.')
    printTime(t_0, n_tabs=2)
    training_error, test_error = 0,0
    Error['training error'].append([])
    Error['test error'].append([])
    layer_rates = []
    for epoch in range(1,num_epochs+1):
        print('\tEpoch %d:'%epoch)
        for i in range(n_train):
            t_batch = time.time()
            [x, y], index = dataset.get_training_batch()
            network.train_batch(x, y, index, b, [learning_rate]*5)
            if i%(int(n_train/20)):
                layer_rates.append(network.get_training_magnitudes())
        # Test training error in the same way as test error
        
        training_error = 0
        t_training_batch_sum = 0
        #*** Testing training error in same manner as test error, to make the two
        #    more comparable
        for i in range(n_train):
            t_batch = time.time()
            [x,y], index = dataset.get_training_batch()
            network.use_persistant_particle(index)
            network.set_x_state(x)
            network.evolve_to_equilibrium(y,0)
            training_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
            t_training_batch_sum += time.time()-t_batch
        t_training_batch_sum /= n_train
        
        test_error = 0
        t_testing_batch_sum = 0
        for i in range(n_test):
            t_batch = time.time()
            [x, y], index = dataset.get_test_batch()
            network.use_persistant_particle(index)
            network.set_x_state(x)
            network.evolve_to_equilibrium(y, 0)
            network.set_persistant_particle(index, network.s)
            test_error += torch.eq(torch.argmax(network.s[:,network.iy],dim=1),torch.argmax(y,dim=1)).sum()
            t_testing_batch_sum += time.time()-t_batch
        t_testing_batch_sum /= n_test
        print('\t\tDone with epoch %d.'%epoch)
        printTime(t_0, n_tabs=2)
        printTime(t_training_batch_sum, msg='Average training batch time: ', n_tabs=2, offset=False)
        printTime(t_testing_batch_sum, msg='Average testing batch time: ', n_tabs=2, offset=False)
        print(('\t\tTraining error: %.06f'%(100*(1-(float(training_error)/n_train_ex))))+'%.'+\
              ('\n\t\tTest error: %.06f'%(100*(1-(float(test_error)/n_test_ex))))+'%.')
        Error['training error'][-1].append(1-(float(training_error)/n_train_ex))
        Error['test error'][-1].append(1-(float(test_error)/n_test_ex))
    Error['layer rates'].append([float(l) for l in layer_rates])
    print('Done training:')
    print(('\tFinal training error: %.06f'%(100*(1-(float(training_error)/n_train_ex))))+'%.')
    print(('\tFinal testing error: %.06f'%(100*(1-(float(test_error)/n_test_ex))))+'%.')
    printTime(t_lr, msg='Total time: ', n_tabs=1, n_nl=1)
    
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












