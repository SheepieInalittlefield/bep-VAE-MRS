import torch
from pythae.models import VAE, AutoModel
from pythae.pipelines import TrainingPipeline
from pythae.trainers.training_callbacks import WandbCallback
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_config, encoder=None, decoder=None):
    model = VAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
    )
    return model


def get_callback(training_config, model_config):
    callbacks = []
    wandb_cb = WandbCallback()
    wandb_cb.setup(
        training_config=training_config,
        model_config=model_config,
        project_name="bep-mia",
        entity_name="sheepieinalittlefield"
    )
    callbacks.append(wandb_cb)
    return callbacks


def get_pipeline(model, training_config):
    pipeline = TrainingPipeline(
        training_config=training_config,
        model=model
    )
    return pipeline


def train(model_path, pipeline, train_dataset, eval_dataset, callback=None):
    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset,
        callbacks=callback
    )
    last_training = sorted(os.listdir(model_path))[-1]
    trained_model = AutoModel.load_from_folder(os.path.join(model_path, last_training, 'final_model'))
    return trained_model


def select_model(getter, parameters):
    from data import load_mrs
    from architecture import MRS_encoder, MRS_decoder
    # from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST

    train_dataset, eval_dataset, ppm = load_mrs()
    config, model_config, path = getter(parameters)
    wandb_callback = get_callback(config, model_config)
    encoder = MRS_encoder(model_config)
    decoder = MRS_decoder(model_config)
    model = get_model(model_config, encoder, decoder)
    pipeline = get_pipeline(model, config)

    return path, pipeline, train_dataset, eval_dataset, wandb_callback, ppm


def train_model(getter, parameters):
    from config import custom_dis_beta_VAE_config, gen_parameters
    from visualization import sample_model, show_generated_data, mse
    from torch import mean

    parameters = gen_parameters(**parameters)
    path, pipeline, train_dataset, eval_dataset, wandb_callback, ppm = select_model(getter, parameters)

    trained_model = train(path, pipeline, train_dataset, eval_dataset, wandb_callback)

    gen_data = sample_model(trained_model)
    eval_data_mean = mean(eval_dataset, dim=0)
    error = mse(gen_data, eval_data_mean)
    return trained_model, float(error)
