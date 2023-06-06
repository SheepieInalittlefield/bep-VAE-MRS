import matplotlib.pyplot as plt
import numpy as np
import pythae.samplers
import torch


def sample_model(trained_model, Sampler=pythae.samplers.NormalSampler, config=None, train_data=None, n=960):
    if config:
        sampler = Sampler(trained_model, config)
        sampler.fit(train_data)
    else:
        sampler = Sampler(trained_model)
    gen_data = sampler.sample(
        num_samples=n
    )
    return gen_data


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
    fig, ax = plt.subplots(nrows=num_rows, ncols=8, figsize=(20,20))
    test_data = test_data.type(torch.float32)
    for i in range(num_rows):
        for j in range(len(x)):
            embedding = model.embed(test_data[:1, :, :])
            embedding[0][i] = x[j]
            gen_data = model.decoder(embedding).reconstruction
            ax[i][j].plot(ppm[0], gen_data[0][0].cpu().detach())
            ax[i][j].invert_xaxis()
            ax[i][j].set_ylim(-1, 0.3)
    plt.show()



def plot_sample(sample, ppm):
    ax = plt.subplot()
    ax.plot(ppm[0], sample.detach()[0])
    ax.invert_xaxis()
    plt.show()


def mse(real_data, gen_data):
    real_data = real_data.cpu()
    mse = np.square(real_data - gen_data).mean()
    return mse


def show_generated_data(gen_data, real_data, target_spectrum, ppm):
    gen_data = gen_data.numpy()
    if type(real_data) != np.ndarray:
        real_data = real_data.numpy()
    ppm = ppm[0]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].plot(ppm, gen_data[0])
    ax[0][1].plot(ppm, target_spectrum)
    ax[1][0].plot(ppm, gen_data[:].mean(axis=0))
    ax[1][1].plot(ppm, real_data[:][0].mean(axis=0))
    ax[0][0].set_title('generated data, single sample')
    ax[0][1].set_title('target spectrum')
    ax[1][0].set_title('generated data, mean of 480 samples')
    ax[1][1].set_title('real data, mean of 960 samples')
    ax[0][0].invert_xaxis()
    ax[0][0].set_xlabel("ppm")
    ax[0][1].invert_xaxis()
    ax[0][1].set_xlabel("ppm")
    ax[1][0].invert_xaxis()
    ax[1][0].set_xlabel("ppm")
    ax[1][1].invert_xaxis()
    ax[1][1].set_xlabel("ppm")
    fig.suptitle("fake vs real comparison")

    plt.show()
