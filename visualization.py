import pythae.samplers
import matplotlib.pyplot as plt
import numpy as np
from random import randint


def sample_model(trained_model):
    normal_sampler = pythae.samplers.NormalSampler(
        model=trained_model
    )

    gen_data = normal_sampler.sample(
        num_samples=1000
    )
    return gen_data


def plot_sample(gen_data):
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

    for i in range(5):
        for j in range(5):
            axes[i][j].imshow(gen_data[i * 5 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)
    plt.show()


def mse(real_data, gen_data):
    real_data = real_data.cpu()
    mse = np.square(real_data - gen_data).mean()
    return mse


def show_generated_data(gen_data, real_data, ppm):
    gen_data = gen_data.numpy()
    if type(real_data) != np.ndarray:
        real_data = real_data.numpy()
    ppm = ppm[0]
    min_ppm = 2.5
    max_ppm = 4

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].plot(ppm, gen_data[0])
    ax[0][1].plot(ppm, real_data)
    ax[1][0].plot(ppm, gen_data[0:200].mean(axis=0))
    ax[1][1].plot(ppm, gen_data.mean(axis=0))
    ax[0][0].set_title('generated data, single sample')
    ax[0][1].set_title('target spectrum')
    ax[1][0].set_title('generated data, mean of 200 samples')
    ax[1][1].set_title('generated data, mean of 1000 samples')
    ax[0][0].invert_xaxis()
    ax[0][0].set_xlabel("ppm")
    #ax[0][0].vlines([min_ppm, max_ppm], -0.35, 0.1, colors="red", linestyles="--", linewidths=1)
    ax[0][1].invert_xaxis()
    ax[0][1].set_xlabel("ppm")
    #ax[0][1].vlines([min_ppm, max_ppm], -0.35, 0.1, colors="red", linestyles="--", linewidths=1)
    ax[1][0].invert_xaxis()
    ax[1][0].set_xlabel("ppm")
    #ax[1][0].vlines([min_ppm, max_ppm], -0.35, 0.1, colors="red", linestyles="--", linewidths=1)
    ax[1][1].invert_xaxis()
    ax[1][1].set_xlabel("ppm")
    #ax[1][1].vlines([min_ppm, max_ppm], -0.35, 0.1, colors="red", linestyles="--", linewidths=1)
    fig.suptitle("fake vs real comparison")

    mse = np.square(real_data - gen_data).mean()

    print(f"MSE: {mse}")

    plt.show()
