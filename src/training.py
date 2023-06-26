import os
import torch
from pythae.models import AutoModel
from pythae.pipelines import TrainingPipeline
from pythae.trainers.training_callbacks import WandbCallback
from src.data import load_mrs_simulations, load_mrs_real
from src.architecture import ConvolutionalEncoder, ConvolutionalDecoder, DenseEncoder, DenseDecoder, \
    DenseDiscriminator
from src.config import gen_parameters
from src.utilities.visualization import sample_model, mse, KLD
from numpy import mean
import time
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_getter, model_config, encoder=None, decoder=None, discriminator=None):
    if discriminator:
        model = model_getter(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )
        model.set_discriminator(discriminator)
    else:
        model = model_getter(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )
    return model


def get_callback(training_config, model_config):
    callbacks = []
    #wandb_cb = WandbCallback()
    #wandb_cb.setup(
    #    training_config=training_config,
    #    model_config=model_config,
    #    project_name="bep-mia",
    #    entity_name="sheepieinalittlefield"
    #)
    #callbacks.append(wandb_cb)
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
    return get_last_trained(model_path)


def get_last_trained(model_path):
    last_training = sorted(os.listdir(model_path))[-1]
    trained_model = AutoModel.load_from_folder(os.path.join(model_path, last_training, 'final_model'))
    return trained_model


def select_model(getter, parameters):
    #if parameters['data'] == 'simulated':
        #train_dataset, eval_dataset, ppm = load_mrs_simulations()
    if parameters['data'] == 'real':
        train_dataset, eval_dataset, test_dataset, ppm = load_mrs_real(averaged=parameters['averaged'], quarter=parameters['quarter'])
    if parameters['data'] == 'augmented':
        train_dataset, eval_dataset, ppm = load_mrs_simulations()
    config, model_config, path, model_getter = getter(parameters)
    wandb_callback = get_callback(config, model_config)
    if parameters['architecture'] == 'convolutional':
        encoder = ConvolutionalEncoder(model_config)
        decoder = ConvolutionalDecoder(model_config)
        discriminator = None
    elif parameters['architecture'] == 'dense':
        encoder = DenseEncoder(model_config)
        decoder = DenseDecoder(model_config)
        discriminator = DenseDiscriminator(model_config) if parameters['disc'] == 'custom' else None
    else:
        encoder = None
        decoder = None
        discriminator = None

    model = get_model(model_getter, model_config, encoder, decoder, discriminator)
    print(model)
    pipeline = get_pipeline(model, config)

    return path, pipeline, train_dataset, eval_dataset, wandb_callback, ppm


def train_model(getter, parameters):
    if parameters['data'] == 'augmented':
        path, pipeline, train_dataset, eval_dataset, wandb_callback, ppm = select_model(getter, parameters)
        trained_model = train(path, pipeline, train_dataset, eval_dataset)  # , wandb_callback)
        parameters['data'] = 'real'
        parameters['epochs'] = parameters['real_epochs']
        path, pipeline, train_dataset, eval_dataset, wandb_callback, ppm = select_model(getter,
                                                                                                      parameters)
        config, _, _, _ = getter(parameters)
        pipeline = get_pipeline(trained_model, config)
        start_time = time.time()
        trained_model = train(path, pipeline, train_dataset, eval_dataset)  # , wandb_callback)
        end_time = time.time()
        training_time = end_time - start_time
        print("Total Time trained: {}".format(training_time))
        error = mse(trained_model, parameters['averaged'])
        kld = KLD(trained_model, parameters['averaged'])
        with open(os.path.join(path, sorted(os.listdir(path))[-1],  'metrics.txt'), 'w') as f:
            f.write(f" mse: {error}, KLD: {kld}, training_time: {training_time}")
        return trained_model, float(error)
    else:
        path, pipeline, train_dataset, eval_dataset, wandb_callback, ppm = select_model(getter, parameters)
        start_time = time.time()
        trained_model = train(path, pipeline, train_dataset, eval_dataset) #, wandb_callback)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Total Time trained: {training_time}")
        error = mse(trained_model, averaged=parameters['averaged'])
        kld = KLD(trained_model, averaged=parameters['averaged']).mean()
        with open(os.path.join(path, sorted(os.listdir(path))[-1],  'metrics.txt'), 'w') as f:
            f.write(f" mse: {error}, KLD: {kld}, training_time: {training_time}")
        return trained_model, float(error), training_time
