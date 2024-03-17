# CS6910-Assignment-1

This README file guides you through all the code present in this repository and provides instructions on how to implement it correctly. <be><be><br>
Also, link to my WandB project: https://wandb.ai/sumanta_roy/CS6910_assignment_1/reports/Sumanta-s-CS6910-Assignment-1--Vmlldzo3MTU4NTE5

## Root Directory

In the main GitHub repository, you will find a main script named `train.py`. This script accepts command-line arguments as directed by the code specifications outlined below. The default parameters are set to Configuration 1 (as described in question number 10), which yielded the best accuracy for the validation dataset of Fashion-MNIST.

### Arguments to be supported:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

The correct way to call the main script script is something like this:

`!python train.py -wp CS6910_assignment_1 -we sumanta_roy -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.001 -m 0.9 -beta 0.5 -beta1 0.9 -beta2 0.999 -eps 0.0001 -w_d 0.0005 -w_i glorot -nhl 3 -sz 128 -a ReLU`

This script trains the neural network and prints the validation accuracy and the test accuracy on either the 'Fashion-MNIST' dataset or the 'MNIST' dataset, using specified hyperparameters. Additionally, the training loss, validation loss, training accuracy, and validation accuracy with increasing steps are logged into WandB.

An example run (one with cross-entropy loss and the other with mean square loss) is shown in the `main_command_line.ipynb` notebook in this root directory.

All the other files are Python scripts that I defined containing functions required for the neural network's operation and training. Let's go over them one by one:

- `activation_functions.py`: This file contains the activation functions and their derivatives, which are called by the backpropagation, forward propagation, and other functions.
- `backward_propagation.py`: This file contains the main function that performs backpropagation. The inputs include the type of loss function, network parameters, input data, labeled data, type of activation function, and the L2 weight decay parameter. The function returns a dictionary containing all the gradients of the loss function with respect to the weights and biases (parameters).
- `forward_propagation.py`: This file contains the main function that performs forward propagation. The function takes the parameters of the network, input data, and type of activation function as inputs, and then returns a dictionary containing the values of the activated neurons for the entire network.
- `initializers.py`: This file contains the initializers for the training process. Two functions are defined - the Glorot (Xavier) initialization and the random initialization.
- `loss_functions.py`: As the name suggests, this file contains the loss functions (both the mean squared loss and the cross-entropy loss).
- `support_functions.py`: This file contains various small functions used in numerous places within the gradient descent algorithm. For example, it includes the one-hot function (used to convert labeled data to one-hot arrays/matrices) and functions for adding, subtracting, multiplying, and dividing entries in a dictionary.
- `updates.py`: This file contains the various update rules for the various gradient descent algorithms.
- `gradient_descent_wandb.py`: This file contains various gradient descent algorithms (mini-batch/stochastic, Adam, Nadam, etc.). Each of these functions calls one of the functions present in the updates.py file.

Now, let's move on to the individual folders containing solutions/answers to individual questions from the assignment. It's important to note that these are small code snippets (for example, a backpropagation code, a forward propagation code, etc.) that use a few functions from the above-defined scripts. Hence, the required Python scripts are called whenever needed. For each question 'x', please open the question_x.ipynb notebook in their respective directories to see the results and the main code.

## Directory: Question 2

This folder contains the notebook cs_6910_question_2.ipynb, where I've created a straightforward forward propagation code. The code is user-friendly and allows us to set the layer architecture (number of hidden layers and number of neurons in each hidden layer), as well as the option to initialize the weights and biases based on a random strategy or the Glorot/Xavier scheme. In this implementation, the forward propagation function takes parameters params, input data X, and an activation function act as inputs. The function calculates the activations for each layer of the neural network using a loop that iterates over the hidden layers. It computes the activations for each layer by applying the corresponding weight matrix Wi and bias vector bi to the previous layer's activations, followed by applying the specified activation function (sigmoid, tanh, or ReLU). The output of each layer is stored in the activations dictionary, which contains both the pre-activation (ai) and post-activation (hi) values. The nomenclature convention is based on Prof. Khapra's lecture notes.

## Directory: Question 3

This folder contains the notebook `cs6910_question_3.ipynb` that contains my implementation of the back-propagation of the feedforward neural network as well as all the gradient-descent algorithms as listed in the question. 

## Directory: Question 4, 8, 10

- Question 4: Bayesian hyperparameter tuning of the cross-entropy-based loss function on the Fashion-MNIST dataset. A sweep was run through the WandB interface.
- Question 8: Bayesian hyperparameter tuning of the mean-squared loss function on the Fashion-MNIST dataset. A sweep was run through the WandB interface.
- Question 10: The three best hyperparameter settings (from the sweep analysis on the Fashion-MNIST dataset) were used to train the model for the MNIST dataset.


