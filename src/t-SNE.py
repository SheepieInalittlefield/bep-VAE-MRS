import numpy as np
import scipy
from data import load_test_real
from visualization import sample_model
from training import get_last_trained
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

embeddings = {
    't-SNE': TSNE(n_components=2, perplexity=50),
    'PCA': PCA(n_components=2)
}
x_real = load_test_real()

model_path = '../custom_VAE'
model = get_last_trained(model_path)
x_fake = sample_model(model, n=2000)

tsne = TSNE(n_components=2, perplexity=50)
x_real = x_real[:,0,:].numpy()
x_fake = x_fake.cpu().numpy()

test = tsne.fit_transform(x_real)
test2 = tsne.fit_transform(x_fake)
fig, ax = plt.subplots()
X1 = MinMaxScaler().fit_transform(test)
X2 = MinMaxScaler().fit_transform(test2)
ax.scatter(X1[:,0],X1[:,1], alpha=0.3, zorder=2)
ax.scatter(X2[:,0],X2[:,1], alpha=0.3, zorder=2)
fig.show()