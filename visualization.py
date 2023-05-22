import pythae.samplers
import matplotlib.pyplot as plt
import numpy as np
from random import randint


def sample_model(trained_model):
    normal_sampler = pythae.samplers.NormalSampler(
        model=trained_model
    )

    gen_data = normal_sampler.sample(
        num_samples=1
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
    gen_data = np.reshape(gen_data, [2048,])
    gen_data = gen_data.numpy()
    real_data = real_data.numpy()
    ppm = ppm[0]
    real_data = real_data[0]

    min_ppm = 2.5
    max_ppm = 4
    max_ind = np.amax(np.where(ppm >= min_ppm))
    min_ind = np.amin(np.where(ppm <= max_ppm))

    x_crop = real_data[min_ind:max_ind]
    #x_crop_norm = (x_crop - x_crop.min()) / (x_crop.max() - x_crop.min())

    y_crop = gen_data[min_ind:max_ind]
    #y_crop_norm = (y_crop - y_crop.min()) / (y_crop.max() - y_crop.min())

    ppm_crop = ppm[min_ind:max_ind]
    print(gen_data.shape)
    print(real_data.shape)
    print(ppm.shape)

    fig, ax = plt.subplots(1,2,figsize=(10, 5))
    ax[0].plot(ppm, gen_data)
    ax[1].plot(ppm, real_data)
    ax[0].set_title('generated data')
    ax[1].set_title('real data (average of eval set)')
    ax[0].invert_xaxis()
    ax[0].set_xlabel("ppm")
    ax[0].set_ylim(-1, 1)
    ax[0].vlines([min_ppm, max_ppm], -1, 1, colors="red", linestyles="--")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("ppm")
    ax[1].set_ylim(-1, 1)
    ax[1].vlines([min_ppm, max_ppm], -1, 1, colors="red", linestyles="--")
    fig.suptitle("fake vs real comparison")

    mse = np.square(real_data - gen_data).mean()

    print(f"MSE: {mse}")

    plt.show()

