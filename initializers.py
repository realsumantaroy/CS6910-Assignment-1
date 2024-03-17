#Random Initialization:
import numpy as np

def init_params(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        W = np.random.rand(layer_sizes[i], layer_sizes[i-1]) - 0.5
        b = np.random.rand(layer_sizes[i], 1) - 0.5
        params['W' + str(i)] = W
        params['b' + str(i)] = b
    return params

#Xavier Initialization:
def xavier_init_params(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i-1]))
        W = np.random.uniform(-limit, limit, size=(layer_sizes[i], layer_sizes[i-1]))
        b = np.zeros((layer_sizes[i], 1))
        params['W' + str(i)] = W
        params['b' + str(i)] = b
    return params
