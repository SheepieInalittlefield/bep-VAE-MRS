import torch.monitor
from pythae.models.nn import BaseEncoder, BaseDecoder, BaseDiscriminator
from pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import torch.nn.functional as funct


class ConvolutionalEncoder(BaseEncoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.n_channels = 1
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Conv1d(self.n_channels, 32, 4, 2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 4, 2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 4, 2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 4, 2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 4, 2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 4, 2, padding=1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),

        )

        self.log_var = nn.Linear(1024, self.latent_dim)
        self.embedding = nn.Linear(1024, self.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h1 = self.layers(x).reshape([x.shape[0], 1024])  # x.shape[0] = batch size
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class ConvolutionalDecoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc = nn.Linear(self.latent_dim, 2048)

        self.layers = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 64, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 1, 3, 2, padding=1, output_padding=1),
            nn.Identity(),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        h1 = self.fc(z).reshape(z.shape[0], 64, 32)
        output = ModelOutput(
            reconstruction=self.layers(h1)
        )
        return output


# not actually dense, cry about it
class DenseEncoder(BaseEncoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.n_channels = 1
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Conv1d(self.n_channels, 2, 3, 2, padding=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 4, 3, 2, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Conv1d(4, 8, 3, 2, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 16, 3, 2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

        )

        self.log_var = nn.Linear(1024, self.latent_dim)
        self.embedding = nn.Linear(1024, self.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h1 = self.layers(x).reshape([x.shape[0], 1024])  # x.shape[0] = batch size
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class DenseDecoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc = nn.Linear(self.latent_dim, 1024)

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.Identity()
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        h1 = self.fc(z).reshape(z.shape[0], 1024, 1)
        output = ModelOutput(
            reconstruction=self.layers(h1)
        )
        return output


class DenseDiscriminator(BaseDiscriminator):
    def __init__(self, args=None):
        BaseDiscriminator.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc = nn.Linear(self.latent_dim, 1024)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h1 = self.fc(x).reshape(x.shape[0], 1024, 1)
        output = ModelOutput(
            adversarial_cost=self.layers(h1)
        )
        return output