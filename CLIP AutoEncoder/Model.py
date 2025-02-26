import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Meno neuroni rispetto a 512
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),  # Riduzione più drastica
            nn.ReLU(),
            nn.Linear(64, 16),  # Bottleneck ancora più stretto
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh(),
        )



    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, inputs):
        en = self.encode(inputs)
        dec = self.decode(en)
        return dec


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=8):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)  # Media
        self.fc_logvar = nn.Linear(64, latent_dim)  # Log-varianza

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, num_classes),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        classification = self.classifier(z)
        return x_recon, mu, logvar, classification