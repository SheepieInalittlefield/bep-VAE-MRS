from config import *
from data import load_target, load_test_real
from src.utilities.visualization import sample_model, mse, show_reconstructions
from training import get_last_trained
from utilities.hyperparameter import *


def main():
    parameters = gen_parameters(
        output_dir='../beta_VAE',
        lr=1e-3,
        batch_size=8,
        epochs=2000,
        dim=64,
        beta=5,
        trainer='base',
        architecture='dense',
        optimizer='AdamW',
        weight_decay=1e-2,
        beta1=0.88,
        beta2=0.99,
        disc='mlp',
        data='real'
    )

    trained_model, mse = train_model(beta_VAE_config, parameters)

    print(f"MSE: {mse}")

    target, ppm = load_target()
    gen_data = sample_model(trained_model).cpu()
    if gen_data.shape[1] == 1:
        gen_data = gen_data[:, 0, :]
    target = target[0]
    test_data = load_test_real()
    show_reconstructions(trained_model, test_data, ppm)


def test_random_stuff():

    model_path = '../linflow_VAE'
    model = get_last_trained(model_path)
    # train, eval, ppm = load_train_real()
    #
    # # target, ppm = load_target()
    # # test_data = load_test_real()
    # gen_data = sample_model(model)
    # #gen_data = gen_data.cpu()
    # # if gen_data.shape[1] == 1:
    # #     gen_data = gen_data[:, 0, :]
    # # target = target.mean(axis=0)
    # # show_generated_data(gen_data, test_data, target, ppm)
    #
    #
    # test_data = load_test_real()
    # target, ppm = load_target()
    # plot_some_generated_samples(gen_data.cpu(), ppm)
    # #plot_static_latent_interpolation(model, test_data, ppm)
    mse(model)



def parameter_search():
    #vae_search()
    #factor_vae_search()
    #info_vae_search()
    IWAE_search()



def get_best_config(path):
    runs = read_from_file(path)
    smallest = [1, 1]
    largest = [0, 0]
    for i in runs:
        if i[1] < smallest[1]:
            smallest = i
        if i[1] > largest[1]:
            largest = i
    print(smallest, largest)


if __name__ == '__main__':
    #get_best_config('custom_VAE1')
    parameter_search()
    #main()
    #test_random_stuff()
