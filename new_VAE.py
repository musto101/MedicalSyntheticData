from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# create a variational autoencoder for synthetic data generation

class encoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim, dropout, activation=nn.Tanh):
        super(encoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        output_dim = 2 * self.latent_dim


        # encoder
        self.encoder = nn.Sequential(
            self.dropout,
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),
            self.dropout,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, x):
        # encode
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, : self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim :]

        return mu, logvar

class decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim, dropout, activation=nn.Tanh):
        super(decoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.output_dim = self.input_dim

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )


    def forward(self, z):
        # decode
        x_hat = self.decoder(z)

        return x_hat

class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim, dropout, activation=nn.Tanh):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.encoder = encoder(latent_dim, input_dim, hidden_dim, dropout, activation)
        self.decoder = decoder(latent_dim, input_dim, hidden_dim, dropout, activation)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        # encode
        mu, logvar = self.encoder(x)

        # reparameterize
        z = self.reparameterize(mu, logvar)

        # decode
        x_hat = self.decoder(z)

        return x_hat, mu, logvar



mci = pd.read_csv('data/mci_preprocessed_wo_csf_vae.csv')

# convert to numpy array
mci_np = mci.values

batch_size = 32
train_loader = torch.tensor(mci_np, dtype=torch.float32)
# define model
model = VAE(latent_dim=2, input_dim=91, hidden_dim=2, dropout=0)

# learning rate anealing
def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)



# Loss function with separate component computations
def loss_function(x_hat, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.functional.cross_entropy(x_hat, x)
    # KL divergence
    epsilon = 1e-10
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar.exp() + epsilon))
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld, recon_loss + kld


# # Training loop
for epoch in range(100000):
    # Forward pass
    # x = torch.tensor(train.values, dtype=torch.float32)
    # x_val = torch.tensor(val.values, dtype=torch.float32)
    x_hat, mu, logvar = model(train_loader)

    # Compute loss components
    recon_loss, kld, total_loss = loss_function(x_hat, train_loader, mu, logvar)

    # Print mu and logvar statistics
    print(
        f"Epoch: {epoch + 1}, mu (min, max, mean): ({mu.min().item()}, {mu.max().item()}, {mu.mean().item()}), logvar (min, max, mean): ({logvar.min().item()}, {logvar.max().item()}, {logvar.mean().item()})")

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print losses
    print(
        f'Epoch: {epoch + 1}, Recon Loss: {recon_loss.item():.4f}, KLD: {kld.item():.4f}, Total Loss: {total_loss.item():.4f}')

#
# # generate new data
z = torch.randn(214, 2)
x_hat = model.decoder(z)
x_hat = x_hat.detach().numpy()
#x_hat = scaler.inverse_transform(x_hat)
x_hat = pd.DataFrame(x_hat, columns=mci.columns)


# save generated data
x_hat.to_csv('data/generated__mci_data.csv')

