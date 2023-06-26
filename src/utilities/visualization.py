import matplotlib.pyplot as plt
import numpy as np
import pythae.samplers
import torch
from random import randint
from data import load_test_real, load_target


def sample_model(trained_model, sampler=pythae.samplers.NormalSampler, config=None, data=None, n=100):
    if config:
        sampler = sampler(trained_model, config)
        sampler.fit(data)
    else:
        sampler = sampler(trained_model)
    gen_data = sampler.sample(
        num_samples=n
    )
    return gen_data


def get_reconstruction(model, data):
    reconstruction = model.predict(data).recon_x
    return reconstruction


def get_embedding(model, data: torch.float32, log_var=False):
    if log_var:
        embeddings, log_var = model.forward(data)
        return embeddings, log_var
    else:
        embeddings = model.embed(data)
        return embeddings


def show_reconstructions(model, real_data, ppm):
    model = model.cpu()
    real_data = real_data.type(torch.float32)
    gen_data = get_reconstruction(model, real_data).detach()
    print(gen_data.shape)
    print(real_data.shape)
    ppm = ppm[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0][0].plot(ppm, gen_data[5])
    ax[0][1].plot(ppm, real_data[5][0])
    ax[1][0].plot(ppm, gen_data[:].mean(axis=0))
    ax[1][1].plot(ppm, real_data[:20].mean(axis=0)[0])
    ax[0][0].set_title('reconstruction')
    ax[0][1].set_title('real data')
    ax[1][0].set_title('reconstructions, averaged')
    ax[1][1].set_title('real data, averaged')
    for i in ax:
        for j in i:
            j.set_ylim(-1, 1)
            j.set_xlim(0, 4)
            j.invert_xaxis()
            j.set_xlabel("ppm")

    fig.suptitle("fake vs real comparison")

    plt.show()


def plot_input_interpolation(trained_model, ppm, start, stop):
    torch.zeros([16, 1, 2048])
    torch.ones([16, 1, 2048])
    interpolation = trained_model.interpolate(torch.zeros([1, 1, 2048]) - 1, torch.ones([1, 1, 2048]) + 1, 25)
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
    for i in range(5):
        for j in range(5):
            gen_data = trained_model.reconstruct(interpolation[0, i * 5 + j])
            ax[i][j].plot(ppm[0], gen_data.detach()[0])
            ax[i][j].invert_xaxis()
    plt.show()


def plot_some_generated_samples(gen_data, ppm):
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    n = 0
    for i in ax:
        for j in i:
            j.plot(ppm[0], gen_data[n])
            j.invert_xaxis()
            j.set_ylim(-1, 1)
            n += 1
    fig.show()
    return 1


def plot_2d_latent_interpolation(model, ppm):
    x = np.linspace(-5, 5, 5)
    y = np.linspace(-5, 5, 5)
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
    for i in range(len(x)):
        for j in range(len(y)):
            embedding = torch.Tensor([[x[i], y[j]]]).to("cuda:0")
            gen_data = model.decoder(embedding).reconstruction
            ax[i][j].plot(ppm[0], gen_data[0][0].cpu().detach())
            ax[i][j].invert_xaxis()
            ax[i][j].set_title(f"x = {x[i]}, y = {y[j]}")
    plt.show()


def plot_static_latent_interpolation(model, test_data, ppm):
    x = np.linspace(-10, 10, 8)
    num_rows = 8
    fig, ax = plt.subplots(nrows=num_rows, ncols=8, figsize=(30, 20))
    test_data = test_data.type(torch.float32).to("cuda:0")
    for i in range(num_rows):
        for j in range(len(x)):
            embedding = model.embed(test_data[:2, :, :])
            embedding[0][i] = x[j]
            gen_data = model.decoder(embedding[:1]).reconstruction
            ax[i][j].plot(ppm[0], gen_data[0].cpu().detach())
            ax[i][j].invert_xaxis()
            ax[i][j].set_ylim(-1, 1)
    plt.show()


def plot_sample(sample, ppm):
    ax = plt.subplot()
    ax.plot(ppm[0], sample.detach()[0])
    ax.invert_xaxis()
    plt.show()


def mse(model, averaged=True, data=None):
    if data == None:
        data = load_test_real(averaged).type(torch.float32)
        reconstruction = get_reconstruction(model, data).detach()
    else:
        reconstruction = get_reconstruction(model, data).detach()
    mse = []
    for i in range(data.shape[0]):
        mse.append(np.square(data[i][0] - reconstruction[i]).mean())
    return np.mean(mse)


def KLD(model, averaged=True, data=None):
    if data == None:
        test_data = load_test_real(averaged).type(torch.float32)
        model_output = model.encoder(test_data)
    else:
        model_output = model.encoder(data)
    mu = model_output.embedding
    log_var = model_output.log_covariance
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    return kld


def show_generated_data(gen_data, real_data, target_spectrum, ppm):
    gen_data = gen_data.numpy()
    if type(real_data) != np.ndarray:
        real_data = real_data.numpy()
    ppm = ppm[0]
    sample = randint(0, 480)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0][0].plot(ppm, gen_data[sample])
    ax[0][1].plot(ppm, real_data[0][0])
    ax[1][0].plot(ppm, gen_data[:].mean(axis=0))
    ax[1][1].plot(ppm, real_data[:40].mean(axis=0)[0])
    ax[0][0].set_title('generated data, single sample')
    ax[0][1].set_title('real data, single sample')
    ax[1][0].set_title('generated data, mean of 480 samples')
    ax[1][1].set_title('real data, mean of 480 samples')
    for i in ax:
        for j in i:
            j.set_ylim(-1, 1)
            j.set_xlim(0, 4)
            j.invert_xaxis()
            j.set_xlabel("ppm")

    fig.suptitle("fake vs real comparison")

    plt.show()
