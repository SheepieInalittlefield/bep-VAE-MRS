from numpy.random import choice, seed
from src.training import train_model
from src.config import VAE_config
import numpy as np
import time
from json import dumps

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def dump_to_file(runs, path):
    path = "../" + path + '.txt'
    print(path)
    with open(path, 'w') as f:
        f.write(dumps(runs, default=np_encoder))


def search(getter, config_dict):
    method = config_dict.pop('method')
    n_runs = config_dict.pop('n_runs')
    output_dir = config_dict.pop('output_dir')

    runs = []
    if method == 'random':
        start = time.time()
        while n_runs:
            parameters = {}
            seed()
            n_runs -= 1
            for i in config_dict:
                parameters[i] = choice(config_dict[i])
            parameters['epochs'] = parameters['epochs'] * parameters['batch_size']
            parameters['output_dir'] = output_dir
            runs.append([parameters, train_model(getter, parameters)[1]])
        end = time.time()
        print(f"total time for search: {end-start}")
    dump_to_file(runs, output_dir)
    return 1


def vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 1,
        'lr': [1e-3, 2e-3, 5e-4],
        'batch_size': [2, 4, 8, 16, 32],
        'epochs': [10],
        'dim': [8, 16, 32, 64],
        'weight_decay': [1e-2, 2e-2, 1e-1, 5e-3],
        'beta1': [0.88, 0.90, 0.92],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['convolutional'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE1',
    }
    ec = search(VAE_config, config_dict)
    return ec

def beta_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 100,
        'lr': [1e-3, 2e-3, 5e-4],
        'batch_size': [2, 4, 8, 16],
        'epochs': [100],
        'dim': [8, 16, 32, 64],
        'beta': [2, 2.5, 3, 3.5, 4],
        'weight_decay': [1e-2, 2e-2, 1e-1, 5e-3],
        'beta1': [0.88, 0.90, 0.92],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'architecture': ['convolutional'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE',
    }
    ec = search(VAE_config, config_dict)
    return ec

def dis_beta_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 100,
        'lr': [1e-3, 2e-3, 5e-4],
        'batch_size': [2, 4, 8, 16],
        'epochs': [100],
        'dim': [8, 16, 32, 64],
        'weight_decay': [1e-2, 2e-2, 1e-1, 5e-3],
        'beta1': [0.88, 0.90, 0.92],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'architecture': ['convolutional'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE',
    }
    ec = search(VAE_config, config_dict)
    return ec

def factor_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 100,
        'lr': [1e-3, 2e-3, 5e-4],
        'batch_size': [2, 4, 8, 16],
        'epochs': [100],
        'dim': [8, 16, 32, 64],
        'weight_decay': [1e-2, 2e-2, 1e-1, 5e-3],
        'beta1': [0.88, 0.90, 0.92],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'architecture': ['convolutional'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE',
    }
    ec = search(VAE_config, config_dict)
    return ec

def linflow_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 100,
        'lr': [1e-3, 2e-3, 5e-4],
        'batch_size': [2, 4, 8, 16],
        'epochs': [100],
        'dim': [8, 16, 32, 64],
        'weight_decay': [1e-2, 2e-2, 1e-1, 5e-3],
        'beta1': [0.88, 0.90, 0.92],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'architecture': ['convolutional'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE',
    }
    ec = search(VAE_config, config_dict)
    return ec

def info_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 100,
        'lr': [1e-3, 2e-3, 5e-4],
        'batch_size': [2, 4, 8, 16],
        'epochs': [100],
        'dim': [8, 16, 32, 64],
        'weight_decay': [1e-2, 2e-2, 1e-1, 5e-3],
        'beta1': [0.88, 0.90, 0.92],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'architecture': ['convolutional'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE',
    }
    ec = search(VAE_config, config_dict)
    return ec
