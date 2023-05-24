from config import *
import numpy as np
def main():
    from visualization import sample_model, show_generated_data
    from training import train_model
    from data import load_mrs_real, load_target
    from torch import mean

    parameters = gen_parameters(
        output_dir='custom_VAE',
        lr=1e-3,
        batch_size=32,
        epochs=400,
        dim=32,
        trainer='base',
        architecture='dense',
        disc='mlp',
        data='real'
    )
    trained_model, mse = train_model(VAE_config, parameters)

    target, ppm = load_target()
    gen_data = sample_model(trained_model)
    gen_data = gen_data.cpu()
    if gen_data.shape[1] == 1:
        gen_data = gen_data[:,0,:]
    target = target.mean(axis=0)
    show_generated_data(gen_data, target, ppm)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def parameter_search():
    from hyperparameter import test
    from json import dumps
    runs = test()
    with open('custom_vae_search.txt', 'w') as f:
        f.write(dumps(runs, default=np_encoder))


if __name__ == '__main__':
    parameter_search()

# MSE: 0.0001698292866272024
# Train loss: 0.9325
# Eval loss: 2.6887
