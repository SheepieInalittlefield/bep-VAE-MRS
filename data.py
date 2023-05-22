import torch.utils.data
import torchvision.datasets as datasets
import h5py
import numpy as np
from torch import mean
from sklearn.preprocessing import normalize


def load_mnist():
    mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None
                                    )
    train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
    return train_dataset, eval_dataset


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, y, ppm):
        super(BasicDataset, self).__init__()

        self.y = y
        self.ppm = ppm

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.y[idx], self.ppm[idx]


def load_mrs_simulations():
    with h5py.File("../MRS_data/sample_data.h5") as hf:
        gt_fids = hf["ground_truth_fids"][()]  # ground truth free induction decay signal value
        ppm = hf["ppm"][()][:1]

    gt_spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)
    gt_diff_spec = np.real(gt_spec[:, :, 1] - gt_spec[:, :, 0])

    y = gt_diff_spec
    y_max = y.max(axis=(1), keepdims=True)
    y_mean = y.mean()
    y = (y - y_mean) / (y_max - y_mean)
    print(y.shape)

    y_train = y[:int(y.shape[0] * 0.8)]
    y_eval = y[int(y.shape[0] * 0.8):]

    y_train = torch.from_numpy(y_train)
    y_eval = torch.from_numpy(y_eval)

    y_train = y_train.reshape(160, 1, 2048)
    y_eval = y_eval.reshape(40, 1, 2048)

    return y_train, y_eval, ppm


def load_mrs_real():
    import matplotlib.pyplot as plt
    with h5py.File("../MRS_data/track_02_training_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]
        ppm = hf['ppm'][()][:1]
    gt_spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)
    gt_diff_spec = np.real(gt_spec[:, :, 1, :] - gt_spec[:, :, 0, :])
    gt_diff_spec = gt_diff_spec[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    y = gt_diff_spec
    y = np.reshape(y, [160,2048])
    y_max = y.max(axis=(0), keepdims=True)
    y_mean = y.mean(axis=0)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.tensor(y)
    print(y.shape)
    y_mean = mean(y, dim=0)
    print(y_mean.shape)
    ax[0].plot(ppm[0], y[0])
    ax[1].plot(ppm[0], y[1])
    ax[0].set_title('generated data')
    ax[1].set_title('real data (average of eval set)')
    ax[0].invert_xaxis()
    ax[0].set_xlabel("ppm")
    ax[0].set_ylim(-1, 1)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("ppm")
    ax[1].set_ylim(-1, 1)
    fig.suptitle("fake vs real comparison")
    plt.show()

    return gt_diff_spec, 2, 3
