# CS6910-Assignment-1

This README file guides you through all the code present in this repository and provides instructions on how to implement it correctly.

## Main Directory

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

```!python train.py -wp CS6910_assignment_1 -we sumanta_roy -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.001 -m 0.9 -beta 0.5 -beta1 0.9 -beta2 0.999 -eps 0.0001 -w_d 0.0005 -w_i glorot -nhl 3 -sz 128 -a relu```

So, this script basically trains the neural network and prints the validation accuracy and the testy accuracy on either the 'Fashion-MNIST' dataset or the 'MNIST' dataset, using a bunch of hyperparameters that we specify. Also, the training loss, validation loss, training accuracy, and validation accuracy with increasing steps are logged into WandB. 

An example run (one of cross-entropy loss and the other for mean square loss) is shown in the ```main_command_line.ipynb``` notebook in this directory itself.

All the other files are python scripts that I defined containing functions that are required for the neural network working and training. Let us go over them one by one:

- ```activation_functions.py```: 


