from numpy.random import choice, seed
from src.training import train_model
from src.config import *
import numpy as np
import time
from json import dumps, loads


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def dump_to_file(runs, path):
    path = "../" + path + '.txt'
    print(path)
    with open(path, 'w') as f:
        f.write(dumps(runs, default=np_encoder))


def read_from_file(path):
    path = "../" + path + '.txt'
    with open(path, 'r') as f:
        runs = f.read()
        runs = loads(runs)
    return runs


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
            for i in config_dict:
                parameters[i] = choice(config_dict[i])
            parameters['epochs'] = parameters['epochs'] * parameters['batch_size']
            parameters['output_dir'] = output_dir
            runs.append([parameters, train_model(getter, parameters)[1]])
            n_runs -= 1
            print(f"run done {n_runs} runs left! \n\n\n\n\n")
        end = time.time()
        print(f"total time for search: {end - start}")
    dump_to_file(runs, output_dir)
    return 1


def vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 500,
        'lr': [1e-3],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [5e-2, 1e-2, 2e-2, 5e-3],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE_params',
    }
    ec = search(VAE_config, config_dict)
    return ec


def beta_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 500,
        'lr': [1e-3],
        'beta': [5],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [1e-2, 2e-2, 5e-3, 5e-2],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'beta_VAE_params',
    }
    ec = search(beta_VAE_config, config_dict)
    return ec


def dis_beta_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 500,
        'lr': [1e-3],
        'beta': [25, 100, 250, 500],
        'C': [5, 10, 20, 50],
        'warmup': [200],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [1e-2, 2e-2, 5e-3, 5e-2],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'dis_beta_VAE_params',
    }
    ec = search(dis_beta_VAE_config, config_dict)
    return ec


def factor_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 500,
        'lr': [1e-3],
        'gamma': [5, 10, 15, 20],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [1e-2, 2e-2, 5e-3, 5e-2],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['adversarial'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'factor_VAE_params',
    }
    ec = search(factor_VAE_config, config_dict)
    return ec


def linflow_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 500,
        'lr': [1e-3],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [1e-2, 2e-2, 5e-3, 5e-2],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'linflow_VAE_params',
    }
    ec = search(linflow_VAE_config, config_dict)
    return ec


def info_vae_search():
    config_dict = {
        'method': 'random',
        'n_runs': 1000,
        'lr': [1e-3],
        'kernel': ['rbf', 'imq'],
        'alpha': [1],
        'lbd': [1000],
        'bandwidth': [1e-2, 1e-1, 1, 2],
        'scales': [None],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [1e-2, 2e-2, 5e-3, 5e-2],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'info_VAE_params',
    }
    ec = search(info_VAE_config, config_dict)
    return ec


def IWAE_search():
    config_dict = {
        'method': 'random',
        'n_runs': 500,
        'lr': [1e-3],
        'n_samples': [5],
        'batch_size': [4, 8, 16, 32],
        'epochs': [80],
        'dim': [64, 128, 256],
        'weight_decay': [1e-2, 2e-2, 5e-3, 5e-2],
        'beta1': [0.86, 0.88, 0.90],
        'beta2': [0.99, 0.999],
        'trainer': ['base'],
        'optimizer': ['AdamW'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'IWAE_params',
    }
    ec = search(IWAE_config, config_dict)
    return ec
