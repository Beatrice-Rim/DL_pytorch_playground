from torch import nn
from copy import deepcopy


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