from numpy.random import choice, seed
from training import train_model
from config import custom_VAE_config
def search(getter, config_dict):

    method = config_dict.pop('method')
    n_runs = config_dict.pop('n_runs')

    runs = []

    if method == 'random':
        while n_runs:
            parameters = {}
            seed()
            n_runs -= 1
            for i in config_dict:
                parameters[i] = choice(config_dict[i])
            runs.append([parameters, train_model(getter, parameters)[1]])
        return runs


def test():
    config_dict = {
        'method': 'random',
        'n_runs': 20,
        'lr': [1e-3, 2e-3, 5e-3, 1e-4, 2e-4, 5e-4, 1e-5, 2e-5, 5e-5],
        'batch_size': [16, 24, 32, 40, 48, 56, 64],
        'epochs': [400],
        'dim': [4, 8, 12, 16, 20, 24, 28, 32],
    }
    runs = search(custom_VAE_config, config_dict)
    return runs