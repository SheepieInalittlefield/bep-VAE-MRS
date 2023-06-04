import pythae.samplers
import torch

from config import *
from utilities.hyperparameter import *
import numpy as np


def main():
    from src.utilities.visualization import sample_model, show_generated_data
    from training import train_model
    from data import load_target

    parameters = gen_parameters(
        output_dir='../custom_VAE',
        lr=2e-3,
        batch_size=16,
        epochs=250,
        dim=32,
        beta=2,
        trainer='base',
        architecture='convolutional',
        optimizer='AdamW',
        weight_decay = 2e-2,
        beta1 = 0.92,
        beta2 = 0.999,
        disc='mlp',
        data='real'
    )
    trained_model, mse = train_model(VAE_config, parameters)
    print(f"MSE: {mse}")
    target, ppm = load_target()
    gen_data = sample_model(trained_model)
    gen_data = gen_data.cpu()
    if gen_data.shape[1] == 1:
        gen_data = gen_data[:, 0, :]
    target = target.mean(axis=0)
    show_generated_data(gen_data, target, ppm)

def test_random_stuff():
    from src.utilities.visualization import plot_latent_interpolation, sample_model, show_generated_data
    from training import get_last_trained
    from data import load_train_real, load_target

    model_path = '../info_VAE'
    model = get_last_trained(model_path)
    target, ppm = load_target()
    gen_data = sample_model(model)
    gen_data = gen_data.cpu()
    if gen_data.shape[1] == 1:
        gen_data = gen_data[:, 0, :]
    target = target.mean(axis=0)
    show_generated_data(gen_data, target, ppm)

    #target, ppm = load_train_real()
    #plot_latent_interpolation(model, ppm)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def parameter_search():
    vae_search()


if __name__ == '__main__':
    parameter_search()
    #main()
    #test_random_stuff()

# MSE: 0.0001698292866272024
# Train loss: 0.9325
# Eval loss: 2.6887
