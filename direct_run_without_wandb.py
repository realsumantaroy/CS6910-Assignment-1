# This is the main file:

#Importing important packages:
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent_wandb import * 


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

X = train_images.reshape(train_images.shape[0], -1).T/225
Y = train_labels
X_test = test_images.reshape(test_images.shape[0], -1).T/225
Y_test = test_labels

validation_ratio = 0.1 #percentage of data for validation
num_validation_samples = int(validation_ratio * X.shape[1])
indices = np.random.permutation(X.shape[1]) #shuffling the indices

validation_indices = indices[:num_validation_samples]
training_indices = indices[num_validation_samples:]

X_train = X[:, training_indices]
Y_train = Y[training_indices]
X_val = X[:, validation_indices]
Y_val = Y[validation_indices]

print("Number of training samples: " + str(X_train.shape[1]))
print("Number of validation samples: " + str(X_val.shape[1]))
print("Number of testing samples: " + str(X_test.shape[1]))


max_epochs=5
no_hidden_layers=3
size_of_hidden_layer=32
weight_decay = 0.
alpha=1e-3
opt='adam' #'sgd','momentum','nesterov','rmsprop','adam','nadam'
batch_size=32
weight='glorot' #glorot,random
activation='relu' #sigmoid,tanh,relu

input_size,output_size = 784,10
layer_sizes = [input_size] + [size_of_hidden_layer] * no_hidden_layers + [output_size]
beta=0.9
epsilon=1e-4
beta1=0.9
beta2=0.999

if opt == 'sgd':
  params = stochastic_gradient_descent(weight, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size)
elif opt == 'momentum':
  params = momentum_gradient_descent(weight, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, beta, layer_sizes, batch_size)
elif opt == 'nesterov':
  params = nesterov_gradient_descent(weight, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, beta, layer_sizes, batch_size)
elif opt == 'rmsprop':
  params = stochastic_gradient_descent_with_RMSProp(weight, activation, weight_decay, X_train, Y_train, X_val, Y_val,max_epochs, alpha, layer_sizes, batch_size, epsilon, beta)
elif opt == 'adam':
  params = stochastic_gradient_descent_with_adam(weight, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size, epsilon, beta1, beta2)
elif opt == 'nadam':
  params = stochastic_gradient_descent_with_nadam(weight, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size, epsilon, beta1, beta2)
else:
  raise ValueError(f"Invalid optimizer option: {opt}")
