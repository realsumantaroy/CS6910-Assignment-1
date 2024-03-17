import argparse
from gradient_descent_wandb import *
from forward_propagation import forward_prop
from activation_functions import *
from keras.datasets import fashion_mnist, mnist
import wandb
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network with specified hyperparameters')
    
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', help='Project name for Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname', help='Wandb Entity for Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default='fashion_mnist', help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train the neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum for momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta for rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 for adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 for adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay for optimizers')
    parser.add_argument('-w_i', '--weight_init', choices=['random', 'Xavier'], default='Xavier', help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a layer')
    parser.add_argument('-a', '--activation', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='tanh', help='Activation function for hidden layers')
    
    args = parser.parse_args()
    return args

def main(args):
    if args.dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    else:
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Preprocessing the data and keeping aside some for validation
    X = train_images.reshape(train_images.shape[0], -1).T / 255
    Y = train_labels
    validation_ratio = 0.1 #percentage of data for validation
    num_validation_samples = int(validation_ratio * X.shape[1])
    indices = np.random.permutation(X.shape[1]) #shuffling the indices
    validation_indices = indices[:num_validation_samples]
    training_indices = indices[num_validation_samples:]
    X_train = X[:, training_indices]
    Y_train = Y[training_indices]
    X_val = X[:, validation_indices]
    Y_val = Y[validation_indices]
    X_test = test_images.reshape(test_images.shape[0], -1).T / 255
    Y_test = test_labels
    
    # Setting up the hyperparameters
    layer_sizes = [X.shape[0]] + [args.hidden_size] * args.num_layers + [len(np.unique(Y))]
    activation = args.activation
    if args.weight_init=='Xavier':
      weight_init = 'glorot'
    else:
      weight_init = args.weight_init
    max_epochs = args.epochs
    alpha = args.learning_rate
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epsilon = args.epsilon
    loss_type = args.loss

    # Initializing and runing WandB
    wandb.login(key='4734e60951ce310dbe17484eeeb5b3366b54850f')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update(args)
    
    k = 50 #frequency of log generation

    # Selecting the optimizer
    if args.optimizer == 'sgd':
        params = stochastic_gradient_descent(loss_type,k,weight_init, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size)
    elif args.optimizer == 'momentum':
        params = momentum_gradient_descent(loss_type,k,weight_init, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, args.momentum, layer_sizes, batch_size)
    elif args.optimizer == 'nag':
        params = nesterov_gradient_descent(loss_type,k,weight_init, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, args.momentum, layer_sizes, batch_size)
    elif args.optimizer == 'rmsprop':
        params = stochastic_gradient_descent_with_RMSProp(loss_type,k,weight_init, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size, epsilon, args.beta)
    elif args.optimizer == 'adam':
        params = stochastic_gradient_descent_with_adam(loss_type,k,weight_init, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size, epsilon, args.beta1, args.beta2)
    elif args.optimizer == 'nadam':
        params = stochastic_gradient_descent_with_nadam(loss_type,k,weight_init, activation, weight_decay, X_train, Y_train, X_val, Y_val, max_epochs, alpha, layer_sizes, batch_size, epsilon, args.beta1, args.beta2)
    else:
        raise ValueError(f"Invalid optimizer option: {args.optimizer}")
      
    wandb.finish()

    #Test accuracy:
    # Accuracy checker:
    act_test = forward_prop(params, X_test, activation)
    test = act_test[f'h{len(layer_sizes)-1}']
    reverse_onehat = np.argmax(test, axis=0)
    test_accuracy = (np.count_nonzero((reverse_onehat-Y_test) == 0)/len(Y_test))*100
    print("Accuracy on test data= " + str(test_accuracy) + " %")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
