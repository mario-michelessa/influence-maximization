import torch

import torch.nn as nn


def make_fc(num_features, num_layers, activation, hidden_layers_sizes, dropout = 0):
    """
    hidden_layers size : list of int of length num_layers-1
    """
    if num_layers > 1:
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid
        elif activation == 'swish':
            activation_fn = nn.SiLU
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [nn.Linear(num_features, hidden_layers_sizes[0]), activation_fn()]
        for hidden in range(num_layers-2):
            net_layers.append(nn.Dropout(dropout))
            net_layers.append(nn.Linear(hidden_layers_sizes[hidden], hidden_layers_sizes[hidden + 1]))
            net_layers.append(activation_fn())
            net_layers.append(nn.Dropout(dropout))
        net_layers.append(nn.Dropout(dropout))
        net_layers.append(nn.Linear(hidden_layers_sizes[-1], 1))
        net_layers.append(nn.Hardtanh(min_val = 0, max_val = 1.0))
        
        return nn.Sequential(*net_layers)
    else:
        return nn.Sequential(nn.Linear(num_features, 1), nn.Hardtanh(min_val = 0, max_val = 1.0))

