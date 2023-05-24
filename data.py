import torch.utils.data
import torchvision.datasets as datasets
import h5py
import numpy as np
from torch.nn.functional import normalize
from sklearn.preprocessing import RobustScaler


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
    with h5py.File("MRS_data/sample_data.h5") as hf:
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


def load_train_real():
    with h5py.File("MRS_data/track_02_training_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]
        ppm = hf["ppm"][()][:1]

    gt_spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)
    gt_diff_spec = np.real(gt_spec[:, :, 1, :] - gt_spec[:, :, 0, :])

    y = gt_diff_spec
    # y_max = y.max(axis=1, keepdims=True)
    # y_mean = y.mean(axis=1, keepdims=True)
    # y = (y - y_mean) / (y_max - y_mean)
    y = y.mean(axis=0)
    y = y.transpose()
    y = np.reshape(y, [160, 1, 2048])
    y = torch.from_numpy(y)  # at this point the shape of y is [160,1,2048]
    y = normalize(y, p=8, dim=2)

    return y, ppm


def load_test_real():
    with h5py.File("MRS_data/track_02_test_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]

    gt_spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)
    gt_diff_spec = np.real(gt_spec[:, :, 1, :] - gt_spec[:, :, 0, :])

    y = gt_diff_spec
    # y_max = y.max(axis=1, keepdims=True)
    # y_mean = y.mean(axis=1, keepdims=True)
    # y = (y - y_mean) / (y_max - y_mean)
    y = y.mean(axis=0)
    y = y.transpose()
    y = np.reshape(y, [40, 1, 2048])
    y = torch.from_numpy(y)  # at this point the shape of y is [40,1,2048]
    y = normalize(y, p=8, dim=2)

    return y


def load_target():
    with h5py.File("MRS_data/track_02_training_data.h5") as hf:
        target = hf["target_spectra"][()]
        ppm = hf["ppm"][()][:1]
    y = target
    y = torch.from_numpy(y)
    y = normalize(y, p=8, dim=1)

    return y, ppm


def load_mrs_real():
    y_train, ppm = load_train_real()
    y_test = load_test_real()

    return y_train, y_test, ppm
