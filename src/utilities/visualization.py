import matplotlib.pyplot as plt
import numpy as np
import pythae.samplers
import torch
from random import randint
from data import load_test_real, load_target

def sample_model(trained_model, Sampler=pythae.samplers.NormalSampler, config=None, train_data=None, n=480):
    if config:
        sampler = Sampler(trained_model, config)
        sampler.fit(train_data)
    else:
        sampler = Sampler(trained_model)
    gen_data = sampler.sample(
        num_samples=n
    )
    return gen_data


def get_reconstruction(model, data):
    reconstruction = model.predict(data).recon_x
    return reconstruction

def get_embedding(model, data: torch.float32):
    embeddings = model.embed(data)
    return embeddings


def show_reconstructions(model, real_data, ppm):
    model = model.cpu()
    real_data = real_data.type(torch.float32)
    gen_data = get_reconstruction(model, real_data).detach()
    ppm = ppm[0]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0][0].plot(ppm, gen_data[3])
    ax[0][1].plot(ppm, real_data[3][0])
    ax[1][0].plot(ppm, gen_data[:].mean(axis=0))
    ax[1][1].plot(ppm, real_data.mean(axis=0)[0])
    ax[0][0].set_title('reconstruction')
    ax[0][1].set_title('real data')
    ax[1][0].set_title('reconstructions, averaged')
    ax[1][1].set_title('real data, averaged')
    for i in ax:
        for j in i:
            j.set_ylim(-1,1)
            j.set_xlim(0,4)
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


def mse(model):
    test_data = load_test_real().type(torch.float32)

    reconstruction = get_reconstruction(model, test_data).detach()
    mse = []
    for i in range(test_data.shape[0]):
        mse.append(np.square(test_data[i][0]-reconstruction[i]).mean())
    return np.mean(mse)


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
    ax[1][1].plot(ppm, real_data.mean(axis=0)[0])
    ax[0][0].set_title('generated data, single sample')
    ax[0][1].set_title('real data, single sample')
    ax[1][0].set_title('generated data, mean of 480 samples')
    ax[1][1].set_title('real data, mean of 480 samples')
    for i in ax:
        for j in i:
            j.set_ylim(-1,1)
            j.set_xlim(0,4)
            j.invert_xaxis()
            j.set_xlabel("ppm")

    fig.suptitle("fake vs real comparison")

    plt.show()
