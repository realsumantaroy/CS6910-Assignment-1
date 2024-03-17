import numpy as np
from activation_functions import *
from support_functions import *

def backward_prop(params, activations, X, Y, act, lambd):
    m = Y.size
    num_layers = len(params) // 2
    one_hot_Y = one_hot(Y)

    gradients = {}
    
    if act == 'sigmoid':
        grad_activation = grad_sigmoid
    elif act == 'relu':
        grad_activation = grad_ReLU
    elif act == 'tanh':
        grad_activation = grad_tanh

    # derivative of MSE loss with respect to output layer activations
    dL_da_last = activations[f'h{num_layers}'] - one_hot_Y

    # Backpropagate through output layer
    dw_last = 1 / m * np.dot(dL_da_last, activations[f'h{num_layers-1}'].T) + (lambd / m) * params[f'W{num_layers}']
    db_last = 1 / m * np.sum(dL_da_last, axis=1, keepdims=True)
    gradients[f'dW{num_layers}'] = dw_last
    gradients[f'db{num_layers}'] = db_last

    # Backpropagate through hidden layers
    dL_dai = dL_da_last
    for i in reversed(range(1, num_layers)):
        dL_dai = np.dot(params[f'W{i+1}'].T, dL_dai) * grad_activation(activations[f'a{i}'])

        dw = 1 / m * np.dot(dL_dai, activations[f'h{i-1}'].T) + (lambd / m) * params[f'W{i}']
        db = 1 / m * np.sum(dL_dai, axis=1, keepdims=True)
        gradients[f'dW{i}'] = dw
        gradients[f'db{i}'] = db

    return gradients
