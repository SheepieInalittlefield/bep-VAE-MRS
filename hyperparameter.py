from numpy.random import choice, seed
from training import train_model
from config import VAE_config


def search(getter, config_dict):
    method = config_dict.pop('method')
    n_runs = config_dict.pop('n_runs')
    output_dir = config_dict.pop('output_dir')

    runs = []

    if method == 'random':
        while n_runs:
            parameters = {}
            seed()
            n_runs -= 1
            for i in config_dict:
                parameters[i] = choice(config_dict[i])
            parameters['output_dir'] = output_dir
            runs.append([parameters, train_model(getter, parameters)[1]])
        return runs


def test():
    config_dict = {
        'method': 'random',
        'n_runs': 1,
        'lr': [1e-2, 1e-3, 2e-3, 5e-3, 1e-4, 2e-4, 5e-4, 5e-5],
        'batch_size': [8, 16, 24, 32, 40, 48, 56, 64],
        'epochs': [200],
        'dim': [8, 12, 16, 20, 24, 28, 32],
        'trainer': ['base'],
        'architecture': ['dense'],
        'disc': ['mlp'],
        'data': ['real'],
        'output_dir': 'custom_VAE',
    }
    runs = search(VAE_config, config_dict)
    return runs
