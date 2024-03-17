import numpy as np
from initializers import *
from forward_propagation import *
from loss_functions import *
from backward_propagation import *
from updates import *
from support_functions import *
import wandb

def stochastic_gradient_descent(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val, iterations, alpha, layer_sizes, batch_size):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    num_batches = len(X.T) // batch_size
    steps=num_batches // k
    
    for j in range(iterations):

      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})
        
        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)
        params = update_params(params, gradients, alpha)
        

      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
    return params

#momentum-gradient-descent
def momentum_gradient_descent(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val, iterations, alpha, beta, layer_sizes, batch_size):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    history_prev = initialize_history(params)
    num_batches = len(X.T) // batch_size
    steps=num_batches // k

    for j in range(iterations):
      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]       
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})
            
                
        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)

        history = add_dicts(gradients, history_prev, scalar2=beta)
        params = update_momentum_params(params, history, alpha)
        history_prev=history

      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
    return params


#nesterov-gradient-descent
def nesterov_gradient_descent(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val,iterations, alpha, beta, layer_sizes, batch_size):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    history_prev = initialize_history(params)
    num_batches = len(X.T) // batch_size
    steps=num_batches // k

    for j in range(iterations):
      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]       
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})

        nesterov_params = nesterov_params_look(params, history_prev, beta*alpha)
        nesterov_activations = forward_prop(nesterov_params, X_batch, activation)
        nesterov_gradients = backward_prop(loss_type,nesterov_params, nesterov_activations, X_batch, Y_batch, activation, weight_decay)
        history = add_dicts(nesterov_gradients, history_prev, scalar2=beta)
        params = update_momentum_params(params, history, alpha)

        history_prev=history
      
      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
    return params


#sto-gradient-descent-with-adagrad
def stochastic_gradient_descent_with_adagrad(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val,iterations, alpha, layer_sizes, batch_size, epsilon_v):
    k=50
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    num_batches = len(X.T) // batch_size
    steps=num_batches // k
    v_old = initialize_history(params)

    for j in range(iterations):

      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})
        

        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)
        gradients_squared = {key: np.square(value) for key, value in gradients.items()}
        v = add_dicts(gradients_squared,v_old)
        params = update_params_adagrad(params, gradients, alpha, v, epsilon_v)
        v_old = v
      
      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
      
    return params


#sto-gradient-descent-with-rmsprop
def stochastic_gradient_descent_with_RMSProp(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val, iterations, alpha, layer_sizes, batch_size, epsilon_v, beta):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    num_batches = len(X.T) // batch_size
    steps=num_batches // k
    v_old = initialize_history(params)

    for j in range(iterations):

      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})

        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)
        gradients_squared = {key: np.square(value) for key, value in gradients.items()}

        v = add_dicts(gradients_squared,v_old,scalar1=(1-beta),scalar2=beta)
        params = update_params_adagrad(params, gradients, alpha, v, epsilon_v)
        v_old = v

      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))  
    return params


#sto-gradient-descent-with-adadelta
def stochastic_gradient_descent_with_AdaDelta(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val, iterations, alpha, layer_sizes, batch_size, epsilon_v, beta):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    num_batches = len(X.T) // batch_size
    steps=num_batches // k
    v_old = initialize_history(params)
    u_old = initialize_history(params)

    for j in range(iterations):

      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})
       

        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)
        gradients_squared = {key: np.square(value) for key, value in gradients.items()}
        v = add_dicts(gradients_squared,v_old,scalar1=(1-beta),scalar2=beta)
        delw = del_w(u_old,v,gradients,epsilon_v)
        params = update_params_adadelta(params, delw)
        u = add_dicts(gradients_squared,u_old,scalar1=(1-beta),scalar2=beta)

        v_old = v
        u_old = u    

      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
    return params



#sto-gradient-descent-with-adam
def stochastic_gradient_descent_with_adam(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val,iterations, alpha, layer_sizes, batch_size, epsilon_v, beta1, beta2):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    num_batches = len(X.T) // batch_size
    steps=num_batches // k
    m_old = initialize_history(params)
    v_old = initialize_history(params)

    for j in range(iterations):

      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        activations = forward_prop(params, X_batch, activation)
        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})
                
        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)
        gradients_squared = {key: np.square(value) for key, value in gradients.items()}
        m = add_dicts(gradients,m_old,scalar1=(1-beta1),scalar2=beta1)
        m_bar = dict_div(m,(1-beta1))
        v = add_dicts(gradients_squared,v_old,scalar1=(1-beta2),scalar2=beta2)
        v_bar = dict_div(v,(1-beta2))
        params = update_params_adam(params, m_bar, v_bar, epsilon_v, alpha)
        v_old = v
        m_old = m      

      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
    return params



#sto-gradient-descent-with-nadam
def stochastic_gradient_descent_with_nadam(loss_type,k,weight, activation, weight_decay, X, Y, X_val, Y_val, iterations, alpha, layer_sizes, batch_size, epsilon_v, beta1, beta2):
    epsilon = 0.
    if weight=='random':
      params = init_params(layer_sizes)
    elif weight=='glorot':
      params = xavier_init_params(layer_sizes)
    else:
      print("Please choose glorot or random initialization")

    num_batches = len(X.T) // batch_size
    steps=num_batches // k
    m_old = initialize_history(params)
    v_old = initialize_history(params)

    for j in range(iterations):

      for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X[:,start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        activations = forward_prop(params, X_batch, activation)

        if i%steps==0:        
            train_loss = compute_loss(loss_type,activations[f'h{len(layer_sizes)-1}'], Y_batch, epsilon, params, weight_decay)
            train_output = activations[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(train_output, axis=0)
            train_accuracy = (np.count_nonzero((reverse_onehat-Y_batch) == 0)/len(Y_batch))*100
            activations_val = forward_prop(params, X_val, activation)
            val_loss = compute_loss(loss_type,activations_val[f'h{len(layer_sizes)-1}'], Y_val, epsilon, params, weight_decay)
            val_output = activations_val[f'h{len(layer_sizes)-1}']
            reverse_onehat = np.argmax(val_output, axis=0)
            val_accuracy = (np.count_nonzero((reverse_onehat-Y_val) == 0)/len(Y_val))*100
            wandb.log({'train_loss': train_loss,'validation_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'epoch': j+1})
        
              
        gradients = backward_prop(loss_type,params, activations, X_batch, Y_batch, activation, weight_decay)
        gradients_squared = {key: np.square(value) for key, value in gradients.items()}
        m = add_dicts(gradients,m_old,scalar1=(1-beta1),scalar2=beta1)
        m_bar = dict_div(m,(1-beta1))
        v = add_dicts(gradients_squared,v_old,scalar1=(1-beta2),scalar2=beta2)
        v_bar = dict_div(v,(1-beta2))
        bracterm = add_dicts(m,gradients,scalar1=beta1,scalar2=1.)
        params = update_params_adam(params, bracterm, v_bar, epsilon_v, alpha)
        v_old = v
        m_old = m

      print("Epoch: " + str(j+1) + "/" + str(iterations) + "; Train Loss: " + str(train_loss) + "; Val Loss: " + str(val_loss))
    return params

