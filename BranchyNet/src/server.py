import copy
import torch


def average_models(client_updates):
    avg_state = copy.deepcopy(client_updates[0])
    for key in avg_state.keys():
        for i in range(1, len(client_updates)):
            avg_state[key] += client_updates[i][key]
        avg_state[key] = torch.div(avg_state[key], len(client_updates))
    return avg_state


def federated_learning(client_updates, global_model):
    avg_state = average_models(client_updates)
    global_model.load_state_dict(avg_state)
    
    return global_model