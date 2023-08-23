from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )

        # for layer in self.encoder:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_uniform_(layer.weight)
        #         torch.nn.init.zeros_(layer.bias)

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

        # for layer in self.decoder:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_uniform_(layer.weight)
        #         torch.nn.init.zeros_(layer.bias)

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


# class VAE(nn.Module):
#     def __init__(self, latent_dim, input_dim, hidden_dim, dropout):
#         super(VAE, self).__init__()
#
#         self.latent_dim = latent_dim
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(dropout)
#
#         # encoder
#         self.encoder = nn.Sequential(
#             self.dropout,
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             self.dropout,
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.latent_dim * 2),
#         )
#
#         for layer in self.encoder:
#             if isinstance(layer, nn.Linear):
#                 torch.nn.init.xavier_uniform_(layer.weight)
#                 torch.nn.init.zeros_(layer.bias)
#
#         # decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(self.latent_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.input_dim),
#         )
#
#         for layer in self.decoder:
#             if isinstance(layer, nn.Linear):
#                 torch.nn.init.xavier_uniform_(layer.weight)
#                 torch.nn.init.zeros_(layer.bias)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         # encode
#         mu_logvar = self.encoder(x)
#         mu = mu_logvar[:, : self.latent_dim]
#         logvar = mu_logvar[:, self.latent_dim :]
#
#         # reparameterize
#         z = self.reparameterize(mu, logvar)
#
#         # decode
#         x_hat = self.decoder(z)
#
#         return x_hat, mu, logvar

# load data
mci = pd.read_csv('data/mci_preprocessed_wo_csf.csv')
codes = {'CN_MCI': 0, 'Dementia': 1}
mci['last_DX'].replace(codes, inplace=True)
mci = mci.drop(['Unnamed: 0'], axis=1)

# convert to float32
mci = mci.astype('float32')

mci.head()

# drop nan values
mci = mci.dropna()

# scale data
scaler = StandardScaler()
x = scaler.fit_transform(mci)


# # split data into train and validation
# train, val = train_test_split(mci, test_size=0.2, random_state=42)

# define model
model = VAE(latent_dim=2, input_dim=91, hidden_dim=2, dropout=0.1)

# learning rate anealing
def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

# define loss function
# def loss_function(x_hat, x, mu, logvar):
#     # reconstruction loss
#     recon_loss = nn.MSELoss(reduction='sum')(x_hat, x)
#
#     epsilon = 1e-10
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar.exp() + epsilon))
#     return recon_loss + kld



# train model
# for epoch in range(100):
#     # forward pass
#     x = torch.tensor(mci.values, dtype=torch.float32)
#     x_hat, mu, logvar = model(x)
#
#     # compute loss
#     loss = loss_function(x_hat, x, mu, logvar)
#
#     # backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # print loss
#     print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')



# Loss function with separate component computations
def loss_function(x_hat, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.functional.cross_entropy(x_hat, x)
    # KL divergence
    epsilon = 1e-10
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar.exp() + epsilon))
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld, recon_loss + kld

x = torch.tensor(x, dtype=torch.float32)

# Training loop
# for epoch in range(100):
#     # Forward pass
#
#     x_hat, mu, logvar = model(x)
#
#     # Compute loss components
#     recon_loss, kld, total_loss = loss_function(x_hat, x, mu, logvar)
#
#     # Backward pass and optimization
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
#
#     # Print losses
#     print(
#         f'Epoch: {epoch + 1}, Recon Loss: {recon_loss.item():.4f}, KLD: {kld.item():.4f}, '
#         f'Total Loss: {total_loss.item():.4f}')

# Training loop
for epoch in range(10000):
    # Forward pass
    # x = torch.tensor(train.values, dtype=torch.float32)
    # x_val = torch.tensor(val.values, dtype=torch.float32)
    x_hat, mu, logvar = model(x)

    # Compute loss components
    recon_loss, kld, total_loss = loss_function(x_hat, x, mu, logvar)

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


# generate new data
z = torch.randn(100, 2)
x_hat = model.decoder(z)
x_hat = x_hat.detach().numpy()
x_hat = scaler.inverse_transform(x_hat)
x_hat = pd.DataFrame(x_hat, columns=mci.columns)
