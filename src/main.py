from config import *
from data import load_target, load_test_real, load_train_real
from utilities.hyperparameter import *
import torch
from pythae.samplers import GaussianMixtureSamplerConfig, GaussianMixtureSampler
from src.utilities.visualization import plot_2d_latent_interpolation, plot_static_latent_interpolation, sample_model, show_generated_data


def main():
    parameters = gen_parameters(
        output_dir='../dis_beta_VAE',
        lr=2e-3,
        batch_size=16,
        epochs=100,
        dim=8,
        beta=250,
        C=20,
        warmup=250,
        trainer='base',
        architecture='convolutional',
        optimizer='AdamW',
        weight_decay=2e-2,
        beta1=0.92,
        beta2=0.999,
        disc='mlp',
        data='real'
    )

    trained_model, mse = train_model(dis_beta_VAE_config, parameters)

    print(f"MSE: {mse}")

    target, ppm = load_target()
    gen_data = sample_model(trained_model).cpu()
    if gen_data.shape[1] == 1:
        gen_data = gen_data[:, 0, :]
    target = target.mean(axis=0)
    test_data = load_test_real()
    show_generated_data(gen_data, test_data, target, ppm)


def test_random_stuff():
    from training import get_last_trained

    model_path = '../dis_beta_VAE'
    model = get_last_trained(model_path)
    # target, ppm = load_target()
    # test_data = load_test_real()
    # gen_data = sample_model(model)
    # gen_data = gen_data.cpu()
    # if gen_data.shape[1] == 1:
    #     gen_data = gen_data[:, 0, :]
    # target = target.mean(axis=0)
    # show_generated_data(gen_data, test_data, target, ppm)
    test_data = load_test_real()
    target, ppm = load_target()
    plot_static_latent_interpolation(model, test_data, ppm)


def parameter_search():
    vae_search()


if __name__ == '__main__':
    #parameter_search()
    main()
    test_random_stuff()

# MSE: 0.0001698292866272024
# Train loss: 0.9325
# Eval loss: 2.6887
