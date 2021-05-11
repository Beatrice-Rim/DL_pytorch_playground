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

def build_nn(additional_features, activation, num_hidden_layers, hidden_dim):
    if additional_features == 'original':
        input_features = 2 
    elif additional_features == 'both':
        input_features = 7
    else:
        input_features = 5
     
    net = nn.Sequential(
        nn.Linear(input_features, hidden_dim), 
        deepcopy(activation),
        *(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            deepcopy(activation)
        ) for _ in range(num_hidden_layers-1)),
        nn.Linear(hidden_dim, 1)
    )
    return net