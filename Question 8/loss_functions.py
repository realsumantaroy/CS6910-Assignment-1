import numpy as np
from support_functions import *

def compute_loss(Y_pred, Y, epsilon, params, lambd):
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)
    mse_loss = np.mean(np.square(Y_pred - one_hot_Y))

    # Computing L2 regularization term
    l2_regularization = 0
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        l2_regularization += np.sum(np.square(params[f'W{i}']))

    l2_regularization *= (lambd / (2 * m))

    total_loss = mse_loss + l2_regularization
    return total_loss

