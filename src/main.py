from config import *
from pythae.models import AutoModel
from data import load_target, load_test_real, load_mrs_simulations, load_train_real
from pythae.samplers import GaussianMixtureSamplerConfig, GaussianMixtureSampler, IAFSampler, IAFSamplerConfig, \
    TwoStageVAESamplerConfig, TwoStageVAESampler
from src.utilities.visualization import sample_model, mse, show_reconstructions, get_reconstruction, get_embedding, KLD, \
    plot_some_generated_samples
from training import get_last_trained
from utilities.hyperparameter import *
from torch import float32
from tSNE import projection_calc, perplexity_calc
import matplotlib.pyplot as plt
import os
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


def main():
    # parameters = gen_parameters(
    #     output_dir='../custom_VAE',
    #     lr=1e-3,
    #     batch_size=64,
    #     epochs=10,
    #     dim=64,
    #     beta=5,
    #     trainer='base',
    #     architecture='dense',
    #     optimizer='AdamW',
    #     weight_decay=0.01,
    #     beta1=0.86,
    #     beta2=0.99,
    #     disc='mlp',
    #     data='real',
    #     averaged = False,
    #     quarter = False
    # )
    parameters = get_best_config('info_VAE_params', data='real', epochs=1000, output_dir='../Models/info_VAE_160_indiv',
                                 averaged=False, quarter=False, lbd=1e-2, alpha=1)
    trained_model, mse, training_time = train_model(info_VAE_config, parameters)

    target, ppm = load_target()
    test_data = load_test_real(averaged=parameters['averaged'])

    print(f"MSE: {mse}")

    show_reconstructions(trained_model, test_data, ppm)

def temp():
    for i in os.listdir('..'):
        if 'params' in i:
            get_best_config(i.split('.')[0])
def report_visuals():
    test_av = load_test_real(averaged=True).type(float32)[:, 0, :]
    test_indiv = load_test_real(averaged=False).type(float32)[:, 0, :]
    _, ppm = load_target()
    mode = input("select mode (1 for reconstruction, 2 for dim_reduction, 3 for generation, 4 for all)")
    try:
        mode = int(mode)
    except ValueError:
        raise 'ValueError, please make sure to only enter integers from the set (1,2,3)'
    if mode == 1 or mode == 4:
        for i in os.listdir('../Models'):
            model_path = os.path.join('../Models', i)
            last_training = sorted(os.listdir(model_path))[-1]
            last_trained_path = os.path.join(model_path, last_training)
            if "indiv" in i:
                test_data = test_indiv
            else:
                test_data = test_av
            model = AutoModel.load_from_folder(os.path.join(last_trained_path, 'final_model'))
            reconstructions = get_reconstruction(model, test_data).detach()

            with open(os.path.join(last_trained_path, 'metrics.txt')) as f:
                metrics = f.read()
                print(f"{model_path[10:]} metrics: {metrics}")
            ax = plt.subplot()
            ax.plot(ppm[0], reconstructions[11])
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, 4)
            ax.invert_xaxis()
            # ax.set_xlabel("ppm")
            plt.savefig(os.path.join(last_trained_path, f'{i}_recon11.png'), dpi=300)
            plt.show()
            ax = plt.subplot()
        for i in range(2):
            test_data = test_indiv
            ax = plt.subplot()
            ax.plot(ppm[0], test_data[i + 11])
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, 4)
            ax.invert_xaxis()
            # ax.set_xlabel("ppm")
            plt.savefig(f"real {i + 11} indiv", dpi=300)
            plt.show()
    if mode == 2 or mode == 4:
        projections = []
        for name in sorted(os.listdir('../Models')):
            print(name)
            model_path = os.path.join('../Models', name)
            #if not ('beta_VAE' in name or 'info_VAE' in name):
            #    continue
            last_training = sorted(os.listdir(model_path))[-1]
            last_trained_path = os.path.join(model_path, last_training)
            model = AutoModel.load_from_folder(os.path.join(last_trained_path, 'final_model'))
            embeddings = get_embedding(model, test_indiv, log_var=False).detach()
            reconstructions = torch.Tensor()
            test_samples = np.arange(0, 960, 50)
            for i in test_samples:
                embedding = embeddings[i]
                embedding = np.repeat(embedding[np.newaxis, :], 50, axis=0)
                noise = np.random.uniform(0, 0.3, embedding.shape)
                embedding = embedding + noise
                embedding = embedding.type(float32)
                reconstruction = model.decoder(embedding).reconstruction
                reconstructions = torch.cat((reconstructions, reconstruction))
            reconstructions = reconstructions.detach()

            label = name.replace('_', ' ')
            if 'beta' in label:
                label = label.replace('beta ', 'β-')
            if 'info' in label:
                label = label.replace('info ', 'Info-')
            if 'indiv' in label:
                label = label.replace('indiv', 'Individual Transients')
            elif '160' in label:
                label = label.replace('160 av', '40x4')
            else:
                label = label.replace('40 av', '40x1')

            projections.append((projection_calc(reconstructions), label))
            # this is stupid, but I only need to run it once anyway.

        fig, ax = plt.subplots(1,1, figsize=(13,10))
        #labels = []
        #for k in projections:
         #   labels.append(k[1])
        #labels = sorted(labels)
        for ind, j in enumerate(projections):
            t = MinMaxScaler().fit_transform(j[0][0][0])
            ax.scatter(t[:, 0], t[:, 1],
                       alpha=0.4,
                       zorder=3,
                       label=j[1],
                       )
        ax.set_title('t-SNE', size=20, linespacing=2)
        #ax[1].set_title('PCA', size=20, linespacing=2)
        # ax[0].axis("off")
        # ax[1].axis("off")
        ax.legend(loc='best', fontsize='large')
        #ax[1].legend(loc='best', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig('../dim_reduction.png', dpi=600)
        fig.show()
        # projections = []
        # test_projections = []
        # for i in sorted(os.listdir('../Models')):
        #     model_path = os.path.join('../Models', i)
        #     last_training = sorted(os.listdir(model_path))[-1]
        #     last_trained_path = os.path.join(model_path, last_training)
        #     if not (i == 'beta_VAE_160_indiv' or i == 'info_VAE_160_indiv'):
        #         continue
        #     if "indiv" in i:
        #         test_data = test_indiv
        #     else:
        #         test_data = test_av
        #     model = AutoModel.load_from_folder(os.path.join(last_trained_path, 'final_model'))
        #     reconstructions = get_reconstruction(model, test_data).detach()
        #     label = i.replace('_', ' ')
        #     # this is stupid but I only need to run it once anyway.
        #     if 'beta' in label:
        #         label = label.replace('beta ', 'β-')
        #     if 'info' in label:
        #         label = label.replace('info ', 'Info-')
        #     if 'indiv' in label:
        #         label = label.replace('indiv', 'Individual Transients')
        #     elif '160' in label:
        #         label = label.replace('160 av', '40x4')
        #     else:
        #         label = label.replace('40 av', '40x1')
        #     projection = projection_calc(reconstructions)
        #     variance = int(sum(projection[1]['PCA'].explained_variance_ratio_)*100)
        #     label = label + f', {variance}% explained variance'
        #     projections.append((projection, label))
        # for i in range(1,2):
        #     # this is also stupid
        #     data = [test_av, test_indiv][i]
        #     projection = projection_calc(data)
        #     label = ['Real Samples averaged', 'Real Samples'][i]
        #     variance = int(sum(projection[1]['PCA'].explained_variance_ratio_)*100)
        #     label = label + f', {variance}% explained variance'
        #     test_projections.append((projection, label))
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # for ind, j in enumerate(projections):
        #     t = MinMaxScaler().fit_transform(j[0][0][0])
        #     p = MinMaxScaler().fit_transform(j[0][0][1])
        #     ax.scatter(t[:, 0], t[:, 1],
        #                   alpha=0.6,
        #                   zorder=3,
        #                   label=j[1].split(",")[0],
        #                   )
        #     # ax[1].scatter(p[:, 0], p[:, 1],
        #     #               alpha=0.6,
        #     #               zorder=2,
        #     #               label=j[1],
        #     #               )
        # # I am fully aware this a stupid way to do this
        # for ind, j in enumerate(test_projections):
        #     t = MinMaxScaler().fit_transform(j[0][0][0])
        #     p = MinMaxScaler().fit_transform(j[0][0][1])
        #     ax.scatter(t[:, 0], t[:, 1],
        #                   alpha=0.6,
        #                   zorder=2,
        #                   label=j[1].split(",")[0],
        #                   )
        #     # ax[1].scatter(p[:, 0], p[:, 1],
        #     #               alpha=0.6,
        #     #               zorder=2,
        #     #               label=j[1],
        #     #               )
        # ax.set_title('t-SNE', size=20, linespacing=2)
        # #ax[1].set_title('PCA', size=20, linespacing=2)
        # # ax[0].axis("off")
        # # ax[1].axis("off")
        # ax.legend(loc='best', bbox_to_anchor=(1,1))
        # #ax[1].legend(loc='best', bbox_to_anchor=(1,1))
        # plt.tight_layout()
        # plt.savefig('../dim_reduction.png', dpi=600)
        # fig.show()
    if mode == 3 or mode == 4:
        for i in os.listdir('../Models'):
            model_path = os.path.join('../Models', i)
            last_training = sorted(os.listdir(model_path))[-1]
            last_trained_path = os.path.join(model_path, last_training)
            model = AutoModel.load_from_folder(os.path.join(last_trained_path, 'final_model'))
            gen_data = sample_model(model, n=960).detach().cpu()
            fig, ax = plt.subplots()
            for i in range(gen_data.shape[0]):
                ax.plot(ppm[0], gen_data[i], alpha=0.3)
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, 4)
            ax.invert_xaxis()
            title = model_path.split('/')[2].replace('_', ' ')
            # this is stupid but I only need to run it once anyway.
            if 'beta' in title:
                title = title.replace('beta ', 'β-')
            if 'info' in title:
                title = title.replace('info ', 'Info-')
            if 'indiv' in title:
                title = title.replace('indiv', 'individual transients')
            elif '160' in title:
                title = title.replace('160 av', '40x4 averaged transients')
            else:
                title = title.replace('40 av', '40x1 averaged transients')
            plt.tight_layout()
            plt.savefig('../' + title, dpi=600)
            fig.show()
        test_data = [test_av, test_indiv]
        for i in range(2):
            fig, ax = plt.subplots()
            for j in range(test_data[i].shape[0]):
                ax.plot(ppm[0], test_data[i][j], alpha=0.3)
                ax.set_ylim(-1, 1)
                ax.set_xlim(0, 4)
                ax.invert_xaxis()
            title = 'real average.png' if i == 0 else 'real individual.png'
            plt.tight_layout()
            plt.savefig('../' + title, dpi=600)
            fig.show()
            print(f'{test_data[i]}')
    return 1

def calculate_metrics():
    train_indiv= load_train_real(average=False, quarter=False)[0].type(float32)
    train_av_160 = load_train_real(average=True, quarter=False)[0].type(float32)
    train_av_40 = load_train_real(average=True, quarter=True)[0].type(float32)
    for i in os.listdir('../Models'):
        model_path = os.path.join('../Models', i)
        last_training = sorted(os.listdir(model_path))[-1]
        last_trained_path = os.path.join(model_path, last_training)
        if "indiv" in i:
            data = train_indiv
        elif "160_av" in i:
            data = train_av_160
        else:
            data = train_av_40
        model = AutoModel.load_from_folder(os.path.join(last_trained_path, 'final_model'))
        kld = KLD(model, data=data)
        square_error = mse(model, data=data)
        print(f'model name: {i}, KLD: {kld.mean()}, MSE:{square_error}')
def calculate_mi(test_data, embeddings):
    mi_values = []
    for i in range(embeddings.shape[1]):
        mi = mutual_info_regression(test_data, embeddings[:, i], n_neighbors=5)
        mi_values.append(mi)
        print(f"run no {i} done!")
    return mi_values


def parameter_search():
    vae_search()
    beta_vae_search()
    info_vae_search()
    IWAE_search()


def test():
    test_data = load_test_real(averaged=False).type(float32)
    perplexity_calc(test_data[:, 0, :])


def get_best_config(path, **kwargs):
    runs = read_from_file(path)
    smallest = [1, 1]
    largest = [0, 0]
    for i in runs:
        if i[1] < smallest[1]:
            smallest = i
        if i[1] > largest[1]:
            largest = i
    print(smallest)
    smallest = smallest[0]
    if kwargs:
        for arg in kwargs:
            smallest[arg] = kwargs[arg]
    return smallest


if __name__ == '__main__':
    # get_best_config('IWAE_params2') #0.01225 vs
    # parameter_search()
    # main()
    report_visuals()  # VAE: 0.0215, beta3-VAE: 0.0218, beta10-VAE:0.0214
    # test()
    # calculate_metrics()
    # temp()
