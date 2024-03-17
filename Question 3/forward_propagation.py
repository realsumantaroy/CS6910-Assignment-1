import numpy as np
from activation_functions import *

def forward_prop(params, X, act):
    num_layers = len(params)//2
    activations = {}
    activations['h0'] = X

    for i in range(1, num_layers): #hidden layers
        Wi = params[f'W{i}']
        bi = params[f'b{i}']
        ai = np.dot(Wi, activations[f'h{i-1}']) + bi
        if act=='sigmoid':
          hi = sigmoid(ai)
        elif act=='tanh':
          hi = tanh(ai)
        elif act=='relu':
          hi = ReLU(ai)

        activations[f'a{i}'] = ai
        activations[f'h{i}'] = hi

    W_last = params[f'W{num_layers}'] #last layer
    b_last = params[f'b{num_layers}']
    a_last = np.dot(W_last, activations[f'h{num_layers-1}']) + b_last
    h_last = softmax_mod(a_last)
    activations[f'a{num_layers}'] = a_last
    activations[f'h{num_layers}'] = h_last

    return activations
