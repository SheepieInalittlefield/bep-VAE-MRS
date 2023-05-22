import torch.monitor
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import torch.nn.functional as funct
class MRS_encoder(BaseEncoder):
    def __init__(self, args=None):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.n_channels = 1
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Conv1d(self.n_channels, 16, 3, 4, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 3, 4, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 3, 4, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 3, 4, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 3, 2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, 3, 2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, 3, 2, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.log_var = nn.Linear(1024, self.latent_dim)
        self.embedding = nn.Linear(1024, self.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h1 = self.layers(x).reshape([x.shape[0], 1, 1024])  # x.shape[0] = batch size
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class MRS_decoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc = nn.Linear(self.latent_dim, 1024*8)
        self.layers = nn.Sequential(
            nn.Conv1d(1024, 512, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 256, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 32, 3, 1, padding=1),
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 8, 3, 1, padding=1),
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 1, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Identity()
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        h1 = self.fc(z).reshape(z.shape[0], 1024, 8)
        output = ModelOutput(
            reconstruction=self.layers(h1)
        )
        return output
