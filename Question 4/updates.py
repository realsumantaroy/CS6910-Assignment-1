import numpy as np

#vanill_update_parameters
def update_params(params, gradients, alpha):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - alpha * gradients[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - alpha * gradients[f'db{i}']
    return updated_params

#adagrad_update_parameters
def update_params_adagrad(params, gradients, alpha, v, epsilon):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - alpha/np.sqrt(v[f'dW{i}']+epsilon) * gradients[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - alpha/np.sqrt(v[f'db{i}']+epsilon) * gradients[f'db{i}']
    return updated_params

#nesterov_update_parameters
def update_nesterov_params(params, history, alpha):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - alpha * history[f'W{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - alpha * history[f'b{i}']
    return updated_params

#adadelta_update_parameters
def update_params_adadelta(params, delw):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] + delw[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] + delw[f'db{i}']
    return updated_params

#momentum_update_parameters
def update_momentum_params(params, history, alpha):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - alpha * history[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - alpha * history[f'db{i}']
    return updated_params

#adam_update_parameters
def update_params_adam(params, m_bar, v_bar, epsilon, alpha):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - alpha/(np.sqrt(v_bar[f'dW{i}'])+epsilon) *   m_bar[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - alpha/(np.sqrt(v_bar[f'db{i}'])+epsilon) *   m_bar[f'db{i}']
    return updated_params

#nadam_update_parameters
def update_params_nadam(params, bracterm, v_bar, epsilon, alpha):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - alpha/(np.sqrt(v_bar[f'dW{i}'])+epsilon) *   bracterm[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - alpha/(np.sqrt(v_bar[f'db{i}'])+epsilon) *   bracterm[f'db{i}']
    return updated_params

#nesterov_look_ahead
def nesterov_params_look(params, u, beta):
    updated_params = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
        updated_params[f'W{i}'] = params[f'W{i}'] - beta * u[f'dW{i}']
        updated_params[f'b{i}'] = params[f'b{i}'] - beta * u[f'db{i}']
    return updated_params

#nesterov_look_ahead
def del_w(u,v,gradients,epsilon):
    updated_delw = {}
    num_layers = len(params) // 2
    for i in range(1, num_layers + 1):
            updated_delw[f'dW{i}'] = - (np.sqrt(u[f'dW{i}']+epsilon)/np.sqrt(v[f'dW{i}']+epsilon)) * gradients[f'dW{i}']
            updated_delw[f'db{i}'] = - (np.sqrt(u[f'db{i}']+epsilon)/np.sqrt(v[f'db{i}']+epsilon)) * gradients[f'db{i}']
    return updated_delw
