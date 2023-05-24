from pythae import trainers, models


def trainer_config(parameters):
    if parameters['trainer'] == 'base':
        config = trainers.BaseTrainerConfig(
            output_dir=parameters['output_dir'],
            learning_rate=parameters['lr'],
            per_device_train_batch_size=parameters['batch_size'],
            per_device_eval_batch_size=parameters['batch_size'],
            num_epochs=parameters['epochs'],  # Change this to train the model a bit more
            optimizer_cls=parameters['optimizer'],
            optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)} if parameters[
                                                                                  'optimizer'] == 'AdamW' else None,
            no_cuda=False,
            seed=17
        )
    elif parameters['trainer'] == 'adversarial':
        config = trainers.AdversarialTrainerConfig(
            output_dir=parameters['output_dir'],
            autoencoder_learning_rate=parameters['lr'],
            discriminator_learning_rate=parameters['lr'],
            per_device_train_batch_size=parameters['batch_size'],
            per_device_eval_batch_size=parameters['batch_size'],
            num_epochs=parameters['epochs'],  # Change this to train the model a bit more
            optimizer_cls=parameters['optimizer'],
            optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)} if parameters[
                                                                                  'optimizer'] == 'AdamW' else None,
            no_cuda=False,
            seed=17
        )
    return config


def VAE_MNIST_config():
    config = trainers.BaseTrainerConfig(
        output_dir='basic_MNIST_VAE',
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=1,  # Change this to train the model a bit more
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)},
        no_cuda=False
    )

    model_config = models.VAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=256,
        reconstruction_loss='bce'
    )
    return config, model_config, config.output_dir, models.VAE


def basic_VAE_config():
    config = trainers.BaseTrainerConfig(
        output_dir='basic_MRS_VAE',
        learning_rate=2e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=250,  # Change this to train the model a bit more
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)},
        no_cuda=False,
        seed=42
    )

    model_config = models.VAEConfig(
        input_dim=(1, 2048),
        latent_dim=64,
        reconstruction_loss='mse',
    )
    return config, model_config, config.output_dir, models.VAE


def VAE_config(parameters):
    config = trainer_config(parameters)

    model_config = models.VAEConfig(
        input_dim=(1, 2048),
        latent_dim=parameters['dim'],
        reconstruction_loss='mse',
    )
    return config, model_config, parameters['output_dir'], models.VAE


def beta_VAE_config(parameters):
    config = trainer_config(parameters)

    model_config = models.BetaVAEConfig(
        input_dim=(1, 2048),
        latent_dim=parameters['dim'],
        reconstruction_loss='mse',
        beta=parameters['beta']
    )
    return config, model_config, parameters['output_dir'], models.BetaVAE


def dis_beta_VAE_config(parameters):
    config = trainer_config(parameters)

    model_config = models.DisentangledBetaVAEConfig(
        input_dim=(1, 2048),
        latent_dim=parameters['dim'],
        reconstruction_loss='mse',
        beta=parameters['beta'],
        C=parameters['C'],
        warmup_epoch=parameters['warmup']
    )
    return config, model_config, parameters['output_dir'], models.DisentangledBetaVAE


def factor_VAE_config(parameters):
    config = trainer_config(parameters)

    model_config = models.FactorVAEConfig(
        input_dim=(1, 2048),
        latent_dim=parameters['dim'],
        reconstruction_loss='mse',
        gamma=parameters['gamma']
    )
    return config, model_config, parameters['output_dir'], models.FactorVAE


def linflow_VAE_config(parameters):
    config = trainer_config(parameters)

    model_config = models.VAE_LinNF_Config(
        input_dim=(1, 2048),
        latent_dim=parameters['dim'],
        reconstruction_loss='mse',
        flows=parameters['flows']
    )
    return config, model_config, parameters['output_dir'], models.VAE_LinNF


def info_VAE_config(parameters):
    config = trainer_config(parameters)

    model_config = models.INFOVAE_MMD_Config(
        input_dim=(1, 2048),
        latent_dim=parameters['dim'],
        reconstruction_loss='mse',
        kernel_choice=parameters['kernel'],
        alpha=parameters['alpha'],
        lbd=parameters['lbd'],
        kernel_bandwidth=parameters['bandwidth'],
        scales=parameters['scales']
    )
    return config, model_config, parameters['output_dir'], models.INFOVAE_MMD


def gen_parameters(lr=2e-4, batch_size=32, epochs=250, optimizer='AdamW', dim=32, **kwargs):
    parameters = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer, 'dim': dim}
    if kwargs:
        for kwarg in kwargs:
            parameters[kwarg] = kwargs[kwarg]
    return parameters


def wandb_config_VAE(wandb_config):
    config = trainers.BaseTrainerConfig(
        output_dir='wandb_sweeps',
        learning_rate=wandb_config['lr'],
        per_device_train_batch_size=wandb_config['batch_size'],
        per_device_eval_batch_size=32,
        num_epochs=3,
        optimzer_cls=wandb_config['optimizer'],
        seed=42
    )
    model_config = models.VAEConfig(
        input_dim=(1, 2048),
        latent_dim=wandb_config['latent_dim'],
        reconstruction_loss='mse',
    )
    return config, model_config, config.output_dir
