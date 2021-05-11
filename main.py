from sklearn.datasets import make_moons, make_blobs, make_circles
import numpy as np
from IPython.display import display
from ipywidgets import interactive
import matplotlib.pyplot as plt
%matplotlib inline
from itertools import cycle
from IPython.display import clear_output
from scipy.special import expit
import torch
from torch import nn
from functools import partial

def main(dataset, additional_features, activation, num_hidden_layers, hidden_dim, n_epochs, lr, run, device):
    if dataset == 'moons':
        create_ds = make_moons
    if dataset == 'blobs':
        create_ds = partial(make_blobs, centers=2)
    if dataset == 'circles':
        create_ds = partial(make_circles, factor=0.5)
    
    X, y = create_ds(n_samples=2000)
    
    if activation == 'sigmoid':
        activation_layer = nn.Sigmoid()
    if activation == 'tanh':
        activation_layer = nn.Tanh()
    if activation == 'relu':
        activation_layer = nn.ReLU()
        
    add = additional_features
    
    if add == 'original':
        X = X
    if add == 'squares':
        X = np.hstack([X, X**2, X[:, [0]]*X[:, [1]]])
    if add == 'sin':
        X = np.hstack([X, np.sin(X)])
    if add == 'both':
        X = np.hstack([X, X**2, X[:, [0]]*X[:, [1]], np.sin(X)])
    
    if run:
        net = build_nn(additional_features, activation_layer, num_hidden_layers, hidden_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        train(net, X, y, additional_features, criterion, optimizer, n_epochs, device)
        
interactive(main, 
            dataset=['moons', 'blobs', 'circles'], 
            additional_features=['original', 'squares', 'sin', 'both'],
            activation=['tanh', 'sigmoid', 'relu'],
            num_hidden_layers=(1, 5),
            hidden_dim=(4, 64, 10),
            n_epochs=(1, 100),
            lr=(1e-4, 1e-2, 5e-4),
            run=False,
            device=['cpu', 'cuda']
           )