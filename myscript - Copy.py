#%%
# Code to train on MNIST Dataset

import numpy as np
import torch
from eqp.model import EQP_Network
from eqp.model import Data, Target_MNIST, MNIST_Data
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import datetime
import time

device = torch.device('cuda:0')
torch.set_default_dtype(torch.float)
dtype = torch.float

seed          = 0
eps           = .5
batch_size    = 20
beta          = 1
total_tau     = [10,2]
num_epochs    = 25
learning_rate = [.1,.05]
n_train = 60000
n_test = 10000

layer_sizes = [784,500,10]

epn = EQP_Network(eps=eps,total_tau=total_tau,batch_size=batch_size,seed=seed,
                  layer_sizes=layer_sizes,device=device)
W_0, W_m_0 = epn.initialize_weight_matrix(layer_sizes, seed=seed, kind='smallworld',
                                          symmetric=True, density=.75, num_swconn=0)
epn.initialize_persistant_particles(n_train)

dataset = MNIST_Data(batch_size, device)

Error = {'Training errors': [],
         'Test errors': []}

training_error = 0
for i in range(int(n_train/batch_size)): # compute training error
    [x, y], _ = dataset.get_training_batch()
    epn.set_x_state(x)
    epn.evolve_to_equilbrium(y,0)
    training_error += torch.eq(torch.argmax(epn.s[:,epn.iy],dim=1),torch.argmax(y,dim=1)).sum()
training_error = 1-(float(training_error)/n_train)
test_error = 0
for i in range(int(n_test/batch_size)): # compute test error
    [x, y], _ = dataset.get_test_batch()
    epn.set_x_state(x)
    epn.evolve_to_equilbrium(y,0)
    test_error += torch.eq(torch.argmax(epn.s[:,epn.iy],dim=1),torch.argmax(y,dim=1)).sum()
test_error = 1-(float(test_error)/n_test)
print('Model initialized.\n  Training error:  %.05f\n  Test error:  %.05f'
      %(training_error,test_error))
Error['Training errors'].append(training_error)
Error['Test errors'].append(test_error)

for epoch in range(1, num_epochs+1):
    for i in range(1,int(n_train/batch_size)+1): # train network
        [x, y], indices = dataset.get_training_batch()
        epn.train_batch(x,y,beta,learning_rate,indices)
    training_error = 0
    for i in range(int(n_train/batch_size)): # compute training error
        [x, y], _ = dataset.get_training_batch()
        epn.set_x_state(x)
        epn.evolve_to_equilbrium(y,0)
        training_error += torch.eq(torch.argmax(epn.s[:,epn.iy],dim=1),torch.argmax(y,dim=1)).sum()
    training_error = 1-(float(training_error)/n_train)
    test_error = 0
    for i in range(int(n_test/batch_size)): # compute test error
        [x, y], _ = dataset.get_test_batch()
        epn.set_x_state(x)
        epn.evolve_to_equilbrium(y,0)
        test_error += torch.eq(torch.argmax(epn.s[:,epn.iy],dim=1),torch.argmax(y,dim=1)).sum()
    test_error = 1-(float(test_error)/n_test)
    print('Epoch %d complete.\n  Training error:  %.05f\n  Test error:  %.05f'
          %(epoch,training_error,test_error))
    Error['Training errors'].append(training_error)
    Error['Test errors'].append(test_error)
    
dt = datetime.datetime.now()
filename = r'MNIST_{}-{}-{}-{}-{}.pickle'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute)
pickle.dump(Error, open(filename,'wb'))

 #%% 
 # plot error rate vs. time
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

font = {'family':'Times New Roman','weight':'normal','size':16}
mpl.rc('font',**font)
cmap = mpl.cm.get_cmap('plasma')

filename = r'C:\Users\jig\Documents\GitHub\equilibrium-propagation\trial_data\MNIST_2019-8-19-18-59.pickle'
Errors = pickle.load(open(filename,'rb'))
training_errors = Errors['Training errors']
test_errors = Errors['Test errors']
l = len(training_errors)

plt.plot(range(l),training_errors,'--',c='b',label='Training error')
plt.plot(range(l),test_errors,c='b',label='Test error')
plt.xlabel('Epoch')
plt.ylabel('Error rate (proportion of digits incorrectly identified)')
plt.title('Error rate during training')
plt.legend()


#%%
#tracking magnitude of weight updates vs number of skip-layer connections
import numpy as np

import torch

from eqp.model import EQP_Network
from eqp.model import Data

from tqdm import tqdm

from matplotlib import pyplot as plt


import pickle
import datetime
import time

device = None
torch.set_default_dtype(torch.float)
dtype = torch.float

seed=1
eps = .5
batch_size = 20
beta = 2e0
total_tau = 10
num_epochs = 3
learning_rate = 1e-1
n_train = int(6e4)
n_test = int(1e4)
layer_skipping_connections = range(0,5000,10)


np.set_printoptions(precision=2)


layer_sizes = [16,15,14,13,12,11,10,9,8,7,6,5]

Corrections = []

for trial in range(3): # number of distinct networks to try
    print('\n\nTesting network {}\n'.format(trial+1))
    correction = []
    seed = trial+20
    epn = EQP_Network(eps=eps, total_tau=total_tau, batch_size=batch_size, seed=seed, 
                  layer_sizes=layer_sizes, device=device)
    #store initial network information so network can be kept identical apart from added connections
    initial_state = epn.randomize_initial_state(batch_size=batch_size)
    initial_weight = np.random.randn(epn.num_neurons, epn.num_neurons)
    initial_weight = torch.from_numpy(initial_weight).float().to(device)
    _, W_mask_i = epn.initialize_weight_matrix(layer_sizes, seed=seed, kind='smallworld',
                                             symmetric=True, density=.75, num_swconn=0)
    W_mask_i = W_mask_i.clone()
    W_mask = W_mask_i.clone()
    dataset = Data(layer_sizes[0],layer_sizes[-1],batch_size,n_train=n_train,n_test=n_test)
    i_m = 0
    for i in range(2): # numbers of added connections to try
        # add new connections without changing existing connections
        c = []
        for k in range(len(layer_sizes)-1): c.append([])
        # set up network with initial parameters, plus an added connection
        epn.W = (initial_weight*epn.W_mask).clone()
        epn.s = initial_state.clone()
        print('\nTesting network with {} layer-skipping connections.'.format(
                .5*(np.count_nonzero(epn.W.data.numpy())-np.count_nonzero(initial_weight*W_mask_i))))
        for j in range(50): # number of epochs over which to train
            adj_m = None # mean correction to weight between each pair of layers
            for k in range(1,int(n_train/batch_size)+1):
                x, y = dataset.get_training_batch()
                adj = epn.train_batch(x,y,beta,learning_rate)
                for l in range(len(c)):
                    c[l].append(adj[l])
            print('\tEpoch {} complete'.format(j+1))
        correction.append(c)
        old_W = epn.W_mask.clone()
        #while torch.norm(old_W-epn.W_mask)==0.:
        _, W_mask_new = epn.initialize_weight_matrix(layer_sizes, seed=seed, kind='smallworld',
                                                     symmetric=True, density=.75, num_swconn=1000)
        W_mask = torch.sign(W_mask_new+W_mask).clone()
        epn.W_mask = W_mask.clone()
    Corrections.append(correction)

dt = datetime.datetime.now()
filename = r'training_{}-{}-{}-{}.pickle'.format(dt.month,dt.day,dt.hour,dt.minute)
pickle.dump(Corrections, open(filename,'wb'))


#%%

import numpy as np
from eqp.model import Data, MNIST_Data

device = torch.device('cpu')

np.set_printoptions(precision=10)

training_persistance = []
training_shuffle = []
test_persistance = []
test_shuffle = []

# verify functionality of dataset
for i in range(1):
    batch_size = 20
    dataset = MNIST_Data(batch_size, device)
    initial_dataset = []
    final_dataset = []
    for i in range(int(n_train/batch_size)):
        initial_dataset.append([t.data.numpy() for t in dataset.get_training_batch()])
    for i in range(int(n_train/batch_size)):
        final_dataset.append([t.data.numpy() for t in dataset.get_training_batch()])
        
    n_exist = 0
    n_pos = 0
    for i in range(int(n_train/batch_size)):
        if all([np.array_equal(initial_dataset[i][j],final_dataset[i][j]) for j in range(len(initial_dataset[i]))]):
            n_pos += 1
            n_exist += 1
        else:
            found_item = 0
            for item in final_dataset:
                if all([np.array_equal(initial_dataset[i][j],item[j]) for j in range(len(initial_dataset[i]))]):
                    n_exist += 1
                    found_item = 1
                    break
    training_persistance.append(float(n_exist)/len(initial_dataset))
    training_shuffle.append(1-(float(n_pos)/len(initial_dataset)))

training_persistance = np.array(training_persistance)
training_shuffle = np.array(training_shuffle)

print('Persistance of samples between epochs in training datasets: {} +/- {}'.format(np.mean(training_persistance),2*np.std(training_persistance)))
print('Displacement of samples between epochs in training datasets: {} +/- {}'.format(np.mean(training_shuffle),2*np.std(training_shuffle)))

#%%
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

font = {'family':'Times New Roman','weight':'normal','size':16}
mpl.rc('font',**font)
cmap = mpl.cm.get_cmap('plasma')

filename = r'C:\Users\jig\Documents\GitHub\equilibrium-propagation\trial_data\training_8-15-14-25.pickle'
Corrections = pickle.load(open(filename,'rb'))

n_avg = 500

for correction in Corrections:
    c_0lsc = correction[0]
    c_1klsc = correction[1]
    for i,layer in zip(range(len(c_0lsc)),c_0lsc):
        l_avg = []
        for j in range(n_avg, len(layer)-n_avg):
            p_avg = 0
            for k in range(j-n_avg, j+n_avg+1):
                p_avg += layer[k]
            p_avg /= (2*n_avg+1)
            l_avg.append(p_avg)
        lts = r'c^{'+str(i)+r'}(n)'
        plt.plot(range(len(l_avg)),l_avg,label='$'+lts+'$',c=cmap(float(i+1)/(len(c_0lsc)+1)))
    del c_0lsc
    plt.title('Layer Correction $c^l(n)$ vs Batch $n$ with No Bypass Connections')
    plt.xlabel('Batch $n$')
    plt.ylabel('Layer correction $c^l(n)$')
    plt.yscale('log')
    plt.legend()
    plt.figure()
    for i,layer in zip(range(len(c_1klsc)),c_1klsc):
        l_avg = []
        for j in range(n_avg, len(layer)-n_avg):
            p_avg = 0
            for k in range(j-n_avg, j+n_avg+1):
                p_avg += layer[k]
            p_avg /= (2*n_avg+1)
            l_avg.append(p_avg)
        lts = r'c^{'+str(i)+r'}(n)'
        plt.plot(range(len(l_avg)),l_avg,label='$'+lts+'$',c=cmap(float(i+1)/(len(c_1klsc)+1)))
    del c_1klsc
    plt.title('Layer Correction $c^l(n)$ vs Batch $n$ with 1000 Bypass Connections (17% connectivity)')
    plt.xlabel('Batch $n$')
    plt.ylabel('Layer correction $c^l(n)$')
    plt.yscale('log')
    plt.legend()
    if Corrections.index(correction)!=2: plt.figure()



#%%
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

font = {'family':'normal','weight':'normal','size':16}
mpl.rc('font',**font)

colors = ['b','g','r','c','m','y','k']

layer_sizes = [16,15,14,13,12,11,10,9,8,7,6,5]
lsc = 0
for i in range(len(layer_sizes)):
    lsc += layer_sizes[i]*sum(layer_sizes[i+2:])
n_avg = 1

filename = r'C:\Users\jig\Documents\GitHub\equilibrium-propagation\trial_data\training_8-9-2-41.pickle'
Error = pickle.load(open(filename, 'rb'))
training_errors = Error['Training error']
test_errors = Error['Test error']
weight_corrections = Error['Mean weight correction']
early_weight_corrections = Error['Initial weight correction']
layer_skipping_connections = Error['Layer-skipping connections']

"""
c = [l[0] for l in weight_corrections[0]]
plt.plot([10*i for i in range(len(c))],c,c=colors[0])
plt.plot([10*i for i in range(len(c))],[(1./len(c))*sum(c) for i in range(len(c))],'--',c=colors[0])
"""
# plot training and test errors w.r.t. number of layer-skipping connections
for network in range(len(training_errors)):
    training_error = training_errors[network]
    test_error = test_errors[network]
    plt.plot([n/lsc for n in layer_skipping_connections],training_error,'--',c=colors[network])
    plt.plot([n/lsc for n in layer_skipping_connections],test_error,c=colors[network])
plt.title('Error vs Number of Layer-Skipping Connections',fontsize=20)
plt.xlabel('Layer-Skipping Connections (rel. to full connectivity)',fontsize=20)
plt.ylabel('Error',fontsize=20)

# plot stdev of magnitude of weight matrix corrections w.r.t. number of layer-skipping connections
plt.figure()
for network in range(len(training_errors)):
    weight_correction = weight_corrections[network]
    weight_correction_stdev = []
    for n in range(n_avg,len(weight_correction)-n_avg):
        stdev_m = 0
        for k in range(n-n_avg,n+n_avg+1):
            m = (1/float(len(weight_correction[k])))*sum(weight_correction[k])
            stdev = ((1/float(len(weight_correction[k])-1))*sum([(c-m)**2 for c in weight_correction[k]]))**.5
            stdev_m += (1./(2*n_avg+1))*stdev
        weight_correction_stdev.append(stdev)
    plt.plot([n/lsc for n in layer_skipping_connections[n_avg:-n_avg]],weight_correction_stdev,c=colors[network])
plt.title('Std. Dev. of Weight Correction Norms vs. Number of Layer-Skipping Connections',fontsize=20)
plt.xlabel('Layer-Skipping Connections (rel. to full connectivity)',fontsize=20)
plt.ylabel('Std. Dev. of Weight Correction Norms',fontsize=20)

# plot normalized stdev of weight corrections
plt.figure()
for network in range(len(training_errors)):
    weight_correction = weight_corrections[network]
    weight_correction_stdev = []
    for n in range(n_avg,len(weight_correction)-n_avg):
        stdev_m = 0
        for k in range(n-n_avg,n+n_avg+1):
            m = (1/float(len(weight_correction[k])))*sum(weight_correction[k])
            stdev = ((1/float(len(weight_correction[k])-1))*sum([(c-m)**2 for c in weight_correction[k]]))**.5
            stdev_m += (1./(2*n_avg+1))*(stdev/m)
        weight_correction_stdev.append(stdev_m)
    plt.plot([n/lsc for n in layer_skipping_connections[n_avg:-n_avg]],weight_correction_stdev,c=colors[network])
plt.title('Normalized Std. Dev. of Weight Correction Norms vs. Number of Layer-Skipping Connections',fontsize=20)
plt.xlabel('Layer-Skipping Connections (rel. to full connectivity)',fontsize=20)
plt.ylabel('Normalized Std. Dev. of Weight Correction Norms',fontsize=20)

"""
# plot early weight corrections
plt.figure()
for network in range(len(training_errors)):
    weight_correction = early_weight_corrections[network]
    weight_correction_stdev = []
    for n in range(len(weight_correction)):
        m = (1/float(len(weight_correction[n])))*sum(weight_correction[n])
        stdev = ((1/float(len(weight_correction[n])-1))*sum([(c-m)**2 for c in weight_correction[n]]))**.5
        weight_correction_stdev.append(stdev)
    plt.plot([n/lsc for n in layer_skipping_connections],weight_correction_stdev,c=colors[network])
plt.title('Early Std. Dev. of Weight Correction Norms vs. Number of Layer-Skipping Connections')
plt.xlabel('Layer-Skipping Connections (rel. to full connectivity)')
plt.ylabel('Std. Dev. of Early Weight Correction Norms')
"""

plt.show()


#%% Thing

import numpy as np

import torch

from eqp.model import EQP_Network
from eqp.model import LinearMatrixDataset
from eqp.model import MNISTDataset

from tqdm import tqdm


import pickle
import datetime
import time
    

#device = torch.device('cuda'); torch.set_default_tensor_type(torch.cuda.FloatTensor) # CUDA = uses the GPU
device = None #  Use the CPU instead
torch.set_default_dtype(torch.float)
dtype = torch.float



seed = 3#2
eps = 0.5
batch_size = 20
beta = 0.1
total_tau = 10
num_epochs = 1

layer_sizes = [10,20,15,10,4]
#layer_sizes = [28*28, 500, 10]

Error = [[],[],[],[]]

for i in range(0,100,20):
    learning_rate = 1e-1#1e0
    e = []
    for j in range(1):
        t_0 = time.time()
        epn = EQP_Network(eps=0.5, total_tau=10, batch_size=batch_size, seed=None, layer_sizes = layer_sizes, device = device)
        W, W_mask = epn.initialize_weight_matrix(layer_sizes, seed = seed, kind = 'smallworld',
                                    symmetric = True, density = 0.75, num_swconn=i)
        epn.randomize_initial_state(batch_size = batch_size)
        
        #dataset = MNISTDataset()
        dataset = LinearMatrixDataset(input_size = epn.layer_sizes[0], output_size = epn.layer_sizes[-1], length = 100000,noise=.01)
        dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)
        
        
        # Test script starts here
        costs = []
        error = []
        min_W = None
        avg_error = .5
        min_error = 1
        adjustments = [i,]
        
        for k in range(1):#range(3):
            learning_rate *= 1e-1#
            print('Training with learning rate {}\n'.format(learning_rate))
            final_error = 1
            num_no_imp = 0
            test_error = 1
            cost = 0
            num_epochs = 0
            adjustments.append([])
            #for epoch in tqdm(range(num_epochs)):
            while num_no_imp<5:#(1+k) and num_epochs<(100 if k>0 else 3):
                for batch_idx, (x_data, y_target) in enumerate(dataloader):
                    print(x_data)
                    x_data, y_target = x_data.to(device), y_target.to(device)
                    adj = epn.train_batch(x_data, y_target, beta, learning_rate)
                    adjustments[k+1].append(adj)
                    if batch_idx % 20 == 0:
                        cost = torch.mean(epn.C(y_target))
                        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        #    epoch, batch_idx * len(x_data), len(dataloader.dataset),
                        #    100. * batch_idx / len(dataloader), cost.item()))
                    if batch_idx % 500 == 0:
                        test_error = dataset.validate(epn, num_samples_to_test = 1000)
                        #print('Validation:  Training error %0.4f / Test error %0.4f' % (train_error, test_error))
                        #n = 0;
                        #for v in [e.item() for e in epn.s.flatten()]:
                        #    if v==0. or v==1.: n += 1
                        #print('{} saturated neurons'.format(n))
                        error.append([batch_idx, test_error])
                        costs.append(cost)
                num_epochs += 1
                if test_error<avg_error:
                    num_no_imp = 0 if num_no_imp<2 else num_no_imp-1
                    avg_error = .5*avg_error+.5*test_error
                else:
                    num_no_imp += 1
                if test_error<min_error:
                    min_error = test_error
                    min_W = epn.W.clone()
                    print('new minimum set')
                print('Epoch: {}\nCost: {}\nAverage error: {}\nTest error: {}\nNo improvement: {}\n'.format(num_epochs,cost,avg_error,test_error, num_no_imp))
            print(adjustments[-1][-1])
            e.append(test_error)
            print('Finished trial {} with {} connections. Time taken: {}sec.'.format(j,i,time.time()-t_0))
            epn.W = min_W
            final_error = dataset.validate(epn, num_samples_to_test = 1000)
            print('Final error: {}\n\n'.format(final_error))
    Error[0].append(i)
    Error[1].append(np.array(e).mean())
    Error[2].append(e)
    Error[3].append(adjustments)
    print('Finished all trials with {} connections. Average final error: {}'.format(i,np.array(e).mean()))
    
dt = datetime.datetime.now()
filename = r'trial{}-{}-{}-{}.p'.format(dt.month,dt.day,dt.hour,dt.minute)
pickle.dump(Error, open(filename,'wb'))

#%%
import pickle
from matplotlib import pyplot as plt

filename = r'C:\Users\jig\Documents\GitHub\equilibrium-propagation\trial_data\trial6-25-11-58.p'
Error = pickle.load(open(filename, 'rb'))
fig, ax = plt.subplots()
w = []
for k in range(len(Error[3])):
    data = Error[3][k]
    for j in range(len(data[1][1])):
        w.append([])
        for epochs in data[1:]:
            for epoch in epochs:
                epoch = [e for e in epoch]
                w[j].append(epoch[j])
        plt.scatter(range(len(w[j])),w[j],label='Layer {}'.format(j),s=5)
    plt.title('{} skip-layer connections'.format(data[0]))
    plt.yscale('log')
    plt.legend()
    plt.ylim([1e-12,1e2])
    plt.figure()
plt.show()


#%% Plot state energies

seed = 2
eps = 0.5
batch_size = 2
beta = 0.1
total_tau = 10
learning_rate = 0.01
num_epochs = 1

layer_sizes = [4, 15, 2]

epn = EQP_Network(eps=0.5, total_tau=10, batch_size=batch_size, seed=None, layer_sizes = layer_sizes, device = device)
W, W_mask = epn.initialize_weight_matrix(layer_sizes, seed = seed, kind = 'fc',
                            symmetric = True, density = 0.75)
epn.randomize_initial_state(batch_size = batch_size)

dataset = LinearMatrixDataset(input_size = epn.layer_sizes[0], output_size = epn.layer_sizes[-1])
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)

x_data, y_target = next(iter(dataloader))
x_data, y_target = x_data.to(device), y_target.to(device)

state_list = []
energy_list = []
epn.set_x_state(x_data)
epn.evolve_to_equilbrium(y_target, beta = 0, state_list = state_list, energy_list = energy_list)
epn.evolve_to_equilbrium(y_target, beta = 0.1, state_list = state_list, energy_list = energy_list)
epn.plot_states_and_energy(state_list, energy_list)

#%%

# Setup MNIST data loader
#data_path = os.path.realpath('./mnist_data/')
#train_dataset = datasets.MNIST(data_path, train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
##                       transforms.Normalize((0.5,), (0.3081,))
#                   ]))
#test_dataset = datasets.MNIST(data_path, train=False, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
##                       transforms.Normalize((0.5,), (0.3081,))
#                   ]))
#train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

#        s,d = ep.convert_dataset_batch(data,target, batch_size)
        
# Run training loop
dataset = MNISTDataset()
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)


costs = []

error = []
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (x_data, y_target) in enumerate(dataloader):
        pass
#        epoch = 1
#        s,d = ep.convert_dataset_batch(x_data, y_target)
        epn.train_batch(x_data, y_target, beta, learning_rate)
        if batch_idx % 20 == 0:
            cost = torch.mean(ep.C(s, d))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cost.item()))
        if batch_idx % 500 == 0:
            train_error = ep.validate(train_dataset, num_samples_to_test = 10000)
            test_error = ep.validate(test_dataset, num_samples_to_test = 10000)
            print('Validation:  Training error %0.1f%% / Test error %0.1f%%' % (train_error, test_error))
            error.append([batch_idx, train_error, test_error])
#        costs.append(cost)

#%%

# See how symmetric the weights are
X = W.squeeze().cpu().numpy().copy()
M = np.array(W_mask.cpu().numpy().copy(), dtype = np.bool)
Y = X - X.T
Z = X / X.T
np.std(X.flatten()[M.flatten()])
np.std(Y.flatten()[M.flatten()])
np.mean(Z.flatten()[M.flatten()])
np.median(Z.flatten()[M.flatten()])
Z.flatten()[M.flatten()]
plt.hist(Z.flatten()[M.flatten()])
max(Z.flatten()[M.flatten()])
np.max(X)
plt.hist(np.clip(Z.flatten()[M.flatten()],-5,5))
plt.hist(np.clip(Z.flatten()[M.flatten()],-5,5), bins = 100)


plt.hist(X.flatten()[M.flatten()], bins = 100)
