import numpy as np
import scipy
import torch
from data import load_test_real
from visualization import sample_model, get_reconstruction
from training import get_last_trained
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA


def perplexity_calc(X):
    perplexity = np.arange(10,500,10)
    divergence = []
    for i in perplexity:
        tsne = TSNE(n_components=2, perplexity=i)
        reduced = tsne.fit_transform(X)
        divergence.append(tsne.kl_divergence_)
    plt.plot(perplexity, divergence)
    plt.show()

def projection_calc(X): # X shape should be n_samples, 2048
    embeddings = {
        't-SNE': TSNE(n_components=2, perplexity=50, n_iter=20000),
        'PCA': PCA(n_components=50)
    }
    projections = []
    #reduced = embeddings['PCA'].fit_transform(X)
    #print(f"Percentage of variance explained: {sum(embeddings['PCA'].explained_variance_ratio_)}")
    #projections.append(embeddings['t-SNE'].fit_transform(reduced))
    for i in embeddings.keys():
        projections.append(embeddings[i].fit_transform(X))
    return projections, embeddings

