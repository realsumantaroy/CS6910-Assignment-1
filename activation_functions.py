import numpy as np

def ReLU(x):
  return np.maximum(0,x)
def grad_ReLU(x):
  return (x>0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def grad_sigmoid(x):
    return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

def tanh(x):
    return np.tanh(x)
def grad_tanh(x):
    return 1 - np.tanh(x)**2

def softmax(x):
  return np.exp(x)/np.sum(np.exp(x))
def softmax_mod(x):
    x_stable = x - np.max(x, axis=0, keepdims=True)
    exp_values = np.exp(x_stable)
    softmax_values = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    return softmax_values
