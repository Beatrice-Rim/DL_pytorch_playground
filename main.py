from sklearn.datasets import make_moons, make_blobs, make_circles
import numpy as np
import torch
from torch import nn

from functools import partial
import argparse

from neuralnet import build_nn
from train import train

parser = argparse.ArgumentParser(description='Launch model training')
parser.add_argument('--dataset', type=str, choices=['moons', 'blobs', 'circles'], default='moons')
parser.add_argument('--additional_features', type=str, choices=['original', 'squares', 'sin', 'both'], default='original')
parser.add_argument('--activation', type=str, choices=['sigmoid', 'tanh', 'relu'], default='sigmoid')
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cpu')


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


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
