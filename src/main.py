import pythae.samplers
import torch

from config import *
import numpy as np


def main():
    from src.utilities.visualization import sample_model, show_generated_data
    from training import train_model
    from data import load_target

    parameters = gen_parameters(
        output_dir='../factor_VAE',
        lr=1e-3,
        batch_size=64,
        epochs=500,
        dim=2,
        gamma=0.7,
        trainer='adversarial',
        architecture='convolutional',
        disc='mlp',
        data='real'
    )
    trained_model, mse = train_model(factor_VAE_config, parameters)

    target, ppm = load_target()
    gen_data = sample_model(trained_model)
    gen_data = gen_data.cpu()
    if gen_data.shape[1] == 1:
        gen_data = gen_data[:, 0, :]
    target = target.mean(axis=0)
    show_generated_data(gen_data, target, ppm)

def test_random_stuff():
    from src.utilities.visualization import plot_input_interpolation, plot_latent_interpolation, plot_sample
    from training import get_last_trained
    from data import load_train_real

    model_path = '../factor_VAE'
    model = get_last_trained(model_path)

    target, ppm = load_train_real()
    plot_latent_interpolation(model, ppm)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def parameter_search():
    from src.utilities.hyperparameter import test
    from json import dumps
    runs = test()
    with open('../custom_vae_search.txt', 'w') as f:
        f.write(dumps(runs, default=np_encoder))


if __name__ == '__main__':
    main()
    test_random_stuff()

# MSE: 0.0001698292866272024
# Train loss: 0.9325
# Eval loss: 2.6887
