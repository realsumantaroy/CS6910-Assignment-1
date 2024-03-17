import numpy as np

#One-hot function
def one_hot(labels, num_classes=10):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_classes, num_samples))
    for i, label in enumerate(labels):
        one_hot_labels[label, i] = 1
    return one_hot_labels

#additing two dictionaries:
def add_dicts(dict1, dict2,scalar1=1,scalar2=1):
    result_dict = {}
    for key in dict1.keys():
        result_dict[key] = scalar1*dict1[key] + scalar2*dict2[key]
    return result_dict

#subtract dicts:
def subtract_dicts(dict1, dict2,scalar1=1,scalar2=1):
    result_dict = {}
    for key in dict1.keys():
        result_dict[key] = scalar1*dict1[key] - scalar2*dict2[key]
    return result_dict

#divide dicts:
def dict_div(dictionary, scalar):
    divided_dict = {}
    for key, array in dictionary.items():
        divided_dict[key] = array / scalar
    return divided_dict

#Convert pr to dpr in all the keys of dictionary
def pr_to_dpr(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            new_key = 'd' + key
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict

#Convert dpr to pr in all the keys of dictionary
def dpr_to_pr(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        if key.startswith('d'):
            new_key = key[1:]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict

#Initializing history to zeros, as in v_-1, m_-1
def initialize_history(params):
    history = {}
    for key, value in params.items():
        history['d' + key] = np.zeros_like(value)
    return history
