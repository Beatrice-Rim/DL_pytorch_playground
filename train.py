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


def train(model, x, y, additional_features, criterion, optimizer, n_epochs, device='cpu'):
    data = torch.FloatTensor(x).to(device)
    labels = torch.FloatTensor(y).to(device)
    dataset = torch.utils.data.TensorDataset(data, labels)
    dl = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=2)
    for epoch in range(n_epochs):
        
        model.train()
        for batch in dl:
            optimizer.zero_grad()
            xs, ys = batch
            pred = model(xs).squeeze()
            loss = criterion(pred, ys)
            loss.backward()
            optimizer.step()
        
        model.eval()
        pred = (model(data).squeeze() >= 0.5).float()
        acc = (pred  == labels).float().mean()
            
        plot_level_lines(model, data[:, :2], labels, additional_features, epoch=epoch, acc=acc)
        plt.show()