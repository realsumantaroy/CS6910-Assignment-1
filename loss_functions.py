import numpy as np
from support_functions import *

def compute_loss(loss_type,Y_pred, Y, epsilon, params, lambd):

    if loss_type=='cross_entropy':
        m = Y.shape[0]
        one_hot_Y = one_hot(Y)
        cross_entropy_loss = -np.mean(one_hot_Y * np.log(Y_pred + epsilon))
    
        # Computing L2 regularization term
        l2_regularization = 0
        num_layers = len(params) // 2
        for i in range(1, num_layers + 1):
            l2_regularization += np.sum(np.square(params[f'W{i}']))
    
        l2_regularization *= (lambd / (2 * m))
    
        total_loss = cross_entropy_loss + l2_regularization
        return total_loss
        
    elif loss_type=='mean_squared_error':
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
        
    else:
      print("Please choose mse or cross_entropy")
