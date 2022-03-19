import torch
import torch.nn as nn

def make_fc(num_features, num_layers, activation, intermediate_size):
    if num_layers > 1:
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [nn.Linear(num_features, intermediate_size), activation_fn()]
        for hidden in range(num_layers-2):
            net_layers.append(nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        net_layers.append(nn.Linear(intermediate_size, 1))
        #net_layers.append(nn.ReLU())
        net_layers.append(nn.Hardtanh(min_val = 0, max_val = 1.0))
        return nn.Sequential(*net_layers)
    else:
        return nn.Sequential(nn.Linear(num_features, 1), nn.Hardtanh(min_val = 0, max_val = 1.0))

