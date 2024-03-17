# CS6910-Assignment-1

This README file guides you through all the codes that are present in this repository and the correct way to implement them. 

## Main directory
In the main GitHub repository, there is a main script termed ```train.py``` which takes commandand line arguments as directed by the code specifications, as given below. The default parameters are that of Configuration 1 (in question number 10) which gave the best accuracy for the validation data set of Fashion-MNIST.

** Arguments to be supported: **

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

```  python train.py -wp CS6910_assignment_1 -we sumanta_roy -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.001 -m 0.9 -beta 0.5 -beta1 0.9 -beta2 0.999 -eps 0.0001 -w_d 0.0005 -w_i glorot -nhl 3 -sz 128 -a relu
```

