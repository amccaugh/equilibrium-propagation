
#%% Thing

import numpy as np
from matplotlib import pyplot as plt

import torch

from eqp.model import EQP_Network
from eqp.model import LinearMatrixDataset
from eqp.model import MNISTDataset

from tqdm import tqdm
    

device = torch.device('cuda'); torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_dtype(torch.float)
dtype = torch.float



seed = 2
eps = 0.5
batch_size = 20
beta = 0.1
total_tau = 10
learning_rate = 1e-2
num_epochs = 1

#layer_sizes = [50, 150, 50]
layer_sizes = [28*28, 500, 10]

epn = EQP_Network(eps=0.5, total_tau=10, batch_size=batch_size, seed=None, layer_sizes = layer_sizes, device = device)
W, W_mask = epn.initialize_weight_matrix(layer_sizes, seed = seed, kind = 'sparse',
                            symmetric = True, density = 0.75)
epn.randomize_initial_state(batch_size = batch_size)

dataset = MNISTDataset()
#dataset = LinearMatrixDataset(input_size = epn.layer_sizes[0], output_size = epn.layer_sizes[-1], length = 100000)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True)


costs = []

error = []
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (x_data, y_target) in enumerate(dataloader):
        x_data, y_target = x_data.to(device), y_target.to(device)
        epn.train_batch(x_data, y_target, beta, learning_rate)
        if batch_idx % 20 == 0:
            cost = torch.mean(epn.C(y_target))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x_data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), cost.item()))
        if batch_idx % 500 == 0:
            train_error = dataset.validate(epn, num_samples_to_test = 1000)
            test_error = dataset.validate(epn, num_samples_to_test = 1000)
            print('Validation:  Training error %0.4f / Test error %0.4f' % (train_error, test_error))
            error.append([batch_idx, train_error, test_error])
        costs.append(cost)


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
