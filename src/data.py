import torch.utils.data
import torchvision.datasets as datasets
import h5py
import numpy as np
from torch.nn.functional import normalize
import os.path as path


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
    y_max = y.max(axis=1, keepdims=True)
    y_mean = y.mean()
    y = (y - y_mean) / (y_max - y_mean)

    y_train = y[:int(y.shape[0] * 0.8)]
    y_eval = y[int(y.shape[0] * 0.8):]

    y_train = torch.from_numpy(y_train)
    y_eval = torch.from_numpy(y_eval)

    y_train = y_train.reshape(160, 1, 2048)
    y_eval = y_eval.reshape(40, 1, 2048)
    return y_train, y_eval, ppm


def pre_process_real(gt_fids, n_fids):
    gt_spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)
    gt_diff_spec = np.real(gt_spec[:, :, 1, :] - gt_spec[:, :, 0, :])

    y = gt_diff_spec[:, :, :n_fids]
    y = y.swapaxes(0,2).swapaxes(1,2)
    y = np.reshape(y, [y.shape[0]*y.shape[1], 1, 2048])
    y_max = y.max(axis=2, keepdims=True)
    y_mean = y.mean(axis=2, keepdims=True)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.from_numpy(y)  # at this point the shape of y is [n_samples ,1,2048]
    return y

def load_train_real():
    with h5py.File("../MRS_data/track_02_training_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]  # shape (12, 2048, 2, 160)
        ppm = hf["ppm"][()][:1]
    y = pre_process_real(gt_fids, 160)
    y_train = y[:1800]
    y_eval = y[1800:]
    return y_train, y_eval, ppm


def load_test_real():
    with h5py.File("../MRS_data/track_02_test_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]
    y = pre_process_real(gt_fids, 40)
    return y


def load_target():
    with h5py.File("../MRS_data/track_02_training_data.h5") as hf:
        target = hf["target_spectra"][()]
        ppm = hf["ppm"][()][:1]

    y = target
    y_max = y.max(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.from_numpy(y)

    return y, ppm


def load_mrs_real():
    y_train, y_eval, ppm = load_train_real()
    y_test = load_test_real()

    return y_train, y_eval, y_test, ppm
