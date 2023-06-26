import torch.utils.data
import torchvision.datasets as datasets
import h5py
import numpy as np
from torch.nn.functional import normalize
import os.path as path
import data_corruption


def load_mrs_simulations():
    with h5py.File("../MRS_data/simulated_ground_truths.h5") as hf:
        gt_fids = hf["ground_truth_fids"][()]  # ground truth free induction decay signal value
        ppm = hf["ppm"][()][:1]
        t = hf['t'][()]
    gt_fids = gt_fids[:960]
    t = t[:960]
    y = pre_process_simul(gt_fids, t)
    y_train = y
    y_eval = y
    return y_train, y_eval, ppm


def pre_process_simul(gt_fids, t, averaged=True):
    # add noise
    tm = data_corruption.TransientMaker(gt_fids, t, transients=40)
    tm.add_random_amplitude_noise(15,5)
    tm.add_random_frequency_noise(5, 5)
    tm.add_random_phase_noise(20, 10)
    fids = tm.fids

    spec = np.fft.fftshift(np.fft.ifft(fids, axis=1), axes=1)
    spec_on = spec[:, :, 0, :40]
    if averaged:
        spec_on = spec_on.mean(axis=2, keepdims=True)
    spec_off = spec[:, :, 1, :40]
    if averaged:
        spec_off = spec_off.mean(axis=2, keepdims=True)

    y = np.real(spec_off - spec_on)
    if averaged:
        y = y.swapaxes(1, 2)
    else:
        y = y.swapaxes(0, 2).swapaxes(1, 2)
        y = np.reshape(y, [y.shape[0] * y.shape[1], 1, 2048])
    y_max = y.max(axis=2, keepdims=True)
    y_mean = y.mean(axis=2, keepdims=True)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.from_numpy(y)  # at this point the shape of y is [12, 1, 2048]
    return y


def pre_process_train(gt_fids, averaged, quarter):
    spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)

    spec_on = spec[:, :, 0, :160]
    if averaged:
        spec_on = np.array(np.split(spec_on, [40, 80, 120], axis=2))
        if quarter:
            spec_on = spec_on[:1]
        spec_on = np.concatenate([i for i in spec_on]).mean(axis=2, keepdims=True)

    spec_off = spec[:, :, 1, :160]
    if averaged:
        spec_off = np.array(np.split(spec_off, [40, 80, 120], axis=2))
        if quarter:
            spec_off = spec_off[:1]
        spec_off = np.concatenate([i for i in spec_off]).mean(axis=2, keepdims=True)

    if not averaged and quarter:
        spec_on = spec_on[:, :, :40]
        spec_off = spec_off[:, :, :40]

    y = np.real(spec_off - spec_on)
    if averaged:
        y = y.swapaxes(1, 2)
        print(y.shape)
    else:
        y = y.swapaxes(0, 2).swapaxes(1, 2)
        y = np.reshape(y, [y.shape[0] * y.shape[1], 1, 2048])
    y_max = y.max(axis=2, keepdims=True)
    y_mean = y.mean(axis=2, keepdims=True)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.from_numpy(y)  # at this point the shape of y is [12, 1, 2048]
    return y


def pre_process_test(gt_fids, averaged):
    spec = np.fft.fftshift(np.fft.ifft(gt_fids, axis=1), axes=1)
    if averaged:
        spec_on = spec[:, :, 0, :40].mean(axis=2, keepdims=True)
        spec_off = spec[:, :, 1, :40].mean(axis=2, keepdims=True)
    else:
        spec_on = spec[:, :, 0, :]
        spec_off = spec[:, :, 1, :]
    y = np.real(spec_off - spec_on)
    if averaged:
        y = y.swapaxes(1, 2)
    else:
        y = y.swapaxes(0, 2).swapaxes(1, 2)
        y = np.reshape(y, [y.shape[0] * y.shape[1], 1, 2048])

    y_max = y.max(axis=2, keepdims=True)
    y_mean = y.mean(axis=2, keepdims=True)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.from_numpy(y)  # at this point the shape of y is [12, 1, 2048]
    return y


def load_train_real(average, quarter):
    with h5py.File("../MRS_data/track_02_training_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]  # shape (12, 2048, 2, 160)
        ppm = hf["ppm"][()][:1]
    y = pre_process_train(gt_fids, average, quarter)
    if average and not quarter:
        y_train = y[:40]
        y_eval = y[40:]
    elif average and quarter:
        y_train = y[:10]
        y_eval = y[10:]
    elif not average:
        y_train = y[:1800]
        y_eval = y[1800:]
    return y_train, y_eval, ppm


def load_test_real(averaged):
    with h5py.File("../MRS_data/track_02_test_data.h5") as hf:
        gt_fids = hf["transient_fids"][()]
    y = pre_process_test(gt_fids, averaged)
    return y


def load_target():
    with h5py.File("../MRS_data/track_02_training_data.h5") as hf:
        target = hf["target_spectra"][()]  # shape (12, 1, 2048)
        ppm = hf["ppm"][()][:1]

    y = target
    y_max = y.max(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    y = (y - y_mean) / (y_max - y_mean)
    y = torch.from_numpy(y)

    return y, ppm


def load_mrs_real(averaged=True, quarter=False):
    y_train, y_eval, ppm = load_train_real(averaged, quarter)
    y_test = load_test_real(averaged)

    return y_train, y_eval, y_test, ppm
