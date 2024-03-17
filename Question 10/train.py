from gradient_descent_wandb import *
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import wandb
from types import SimpleNamespace
import random

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

wandb.login(key='4734e60951ce310dbe17484eeeb5b3366b54850f')

sweep_config = {
    'method': 'bayes',
    'name' : 'sweep cross entropy computer',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
         'hidden_size':{
            'values':[32,64,128]
        },
        'hidden_layers':{
            'values':[3,4,5]
        },
        'activation': {
            'values': ['sigmoid','relu','tanh']
        },
        'loss': {
            'values': ['cross_entropy']
        },
        'weight_decay': {
            'values': [0,0.0005,0.5]
        },
        'optimizer': {
            'values': ['sgd','momentum','nesterov','rmsprop','adam','nadam']
        },
        'lr': {
            'values': [1e-3,1e-4]
        },
        'batch_size': {
            'values': [16,32,64] 
        },
        'w_init': {
            'values':['glorot','random'] 
        },
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='CS6910_assignment_1')
wandb.init(project='CS6910_assignment_1', entity='sumanta_roy')

def main():
    '''
    WandB calls main function each time with differnet combination.

    We can retrive the same and use the same values for our hypermeters.

    '''

    with wandb.init() as run:

        run_name="-ac_"+wandb.config.activation+" -hs"+str(wandb.config.hidden_size)+" -hl"+str(wandb.config.hidden_layers)+" -epch"+str(wandb.config.epochs)+" -bs"+str(wandb.config.batch_size)+" -init"+str(wandb.config.w_init)+" -init"+str(wandb.config.lr)+" -l2d"+str(wandb.config.weight_decay)
        wandb.run.name=run_name
        #obj=NN(wandb.config['num_layers'],wandb.config['hidden_size'])

        max_epochs=wandb.config.epochs
        no_hidden_layers=wandb.config.hidden_layers
        size_of_hidden_layer=wandb.config.hidden_size
        weight_decay = wandb.config.weight_decay
        alpha=wandb.config.lr
        opt=wandb.config.optimizer #'sgd','momentum','nesterov','rmsprop','adam','nadam'
        batch_size=wandb.config.batch_size
        weight=wandb.config.w_init #glorot,random
        activation=wandb.config.activation #sigmoid,tanh,relu

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

wandb.agent(sweep_id, function=main,count=100) # calls main function for count number of times.
wandb.finish()
