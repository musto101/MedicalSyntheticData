import pyswarms as ps # import pyswarms
import numpy as np # for manipulating arrays
import pandas as pd # for reading data
import torch # for creating the VAE
from torch import nn, optim # for creating the VAE


class encoder(nn.Module): # define encoder
    def __init__(self, latent_dim, input_dim, hidden_dim, dropout, activation=nn.Tanh):# define init function
        super(encoder, self).__init__() # inherit from nn.Module

        self.latent_dim = latent_dim # define latent dim
        self.input_dim = input_dim # define input dim
        self.hidden_dim = hidden_dim # define hidden dim
        self.dropout = nn.Dropout(dropout) # define dropout
        self.activation = activation # define activation
        output_dim = 2 * self.latent_dim #


        # encoder
        self.encoder = nn.Sequential( # define encoder
            self.dropout,
            nn.Linear(self.input_dim, self.hidden_dim), # define linear layer
            activation(), # define activation
            self.dropout, # define dropout
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, x): # define forward function
        # encode
        mu_logvar = self.encoder(x) #
        mu = mu_logvar[:, : self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim :]

        return mu, logvar

class decoder(nn.Module): # define decoder
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

    def reparameterize(self, mu, logvar): # reparameterize so that we can backpropagate through the model
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
def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


train = pd.read_csv('data/cn_preprocessed_wo_csf_vae.csv').values # read in data
# create validation set
val = train[0:50]

# remove validation set from training set
train = train[50:]
train_loader = torch.tensor(train, dtype=torch.float32) # create train loader
val_loader = torch.tensor(val, dtype=torch.float32) # create val loader


def enforce_hyperparameter_constraints(particle):
    """
    Adjusts the particle's position to ensure all hyperparameters are within valid ranges.

    Args:
    particle (np.ndarray): A single particle representing a set of hyperparameters.
                           Structure: [latent_dim, input_dim, hidden_dim, dropout]

    Returns:
    np.ndarray: The adjusted particle with all hyperparameters within valid ranges.
    """

    # Define the minimum and maximum values for each hyperparameter
    min_latent_dim, max_latent_dim = 1, 50  # Example range for latent_dim
    min_input_dim, max_input_dim = 91, 91  # Fixed value for input_dim based on your dataset
    min_hidden_dim, max_hidden_dim = 50, 500  # Example range for hidden_dim
    min_dropout, max_dropout = 0.0, 0.5  # Example range for dropout rate
    min_l1, max_l1 = 0.0, 1.0
    min_l2, max_l2 = 0.0, 1.0

    # Enforce constraints for latent_dim
    if particle[0] < min_latent_dim:
        particle[0] = min_latent_dim
    elif particle[0] > max_latent_dim:
        particle[0] = max_latent_dim

    # Enforce constraints for input_dim (assuming this is fixed for your dataset)
    particle[1] = max_input_dim  # or set to min_input_dim if they're the same

    # Enforce constraints for hidden_dim
    if particle[2] < min_hidden_dim:
        particle[2] = min_hidden_dim
    elif particle[2] > max_hidden_dim:
        particle[2] = max_hidden_dim

    # Enforce constraints for dropout
    if particle[3] < min_dropout:
        particle[3] = min_dropout
    elif particle[3] > max_dropout:
        particle[3] = max_dropout

    # Enforce constraints for l1
    if particle[4] < min_l1:
        particle[4] = min_l1
    elif particle[4] > max_l1:
        particle[4] = max_l1

    # Enforce constraints for l2
    if particle[5] < min_l2:
        particle[5] = min_l2
    elif particle[5] > max_l2:
        particle[5] = max_l2

    # Ensure the latent and hidden dimensions are integers
    particle[0] = int(round(particle[0]))
    particle[2] = int(round(particle[2]))

    # Ensure the dropout rate is a float
    particle[3] = float(particle[3])

    # Ensure the l1 and l2 are floats
    particle[4] = float(particle[4])
    particle[5] = float(particle[5])

    return particle



def objective_function(params, train_loader=train_loader, epochs=100, val_loader=val_loader):

    # Extract hyperparameters from the PSO particle
    # print(f"Received hyperparameters: {params}")
    latent_dim, input_dim, hidden_dim, dropout, l1, l2 = params
    latent_dim = latent_dim.astype(int)
    hidden_dim = hidden_dim.astype(int)
    input_dim = input_dim.astype(int)
    dropout = dropout.astype(float)
    l1 = l1.astype(float)
    l2 = l2.astype(float)

    # Use hyperparameters to create VAE model
    model = VAE(latent_dim=latent_dim, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, activation=nn.Tanh)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)  # Adjust learning rate as needed

    # Training Loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Check if the batch is just a tensor or a tuple/list of tensors
            if isinstance(batch, torch.Tensor):
                # Handle the case where the batch is just a single tensor (presumably the data)
                x = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Handle the normal case where batch is a tuple of (data, target)
                x, _ = batch
            else:
                raise ValueError(f"Unexpected batch structure: {batch}")
        # reshape data to correct shape
        x = x.view(-1, 91)

        optimiser.zero_grad()
    #
    #         # Forward pass
        x_hat, mu, logvar = model(x)

        loss = loss_function(x_hat, x, mu, logvar, l1, l2, model=model)
        total_loss += loss.item()
        avg_loss = total_loss / 502
        print(f"Training Loss: {avg_loss}")

            # Backward pass
        loss.backward()
        optimiser.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Check if the batch is just a tensor or a tuple/list of tensors
            if isinstance(batch, torch.Tensor):
                # Handle the case where the batch is just a single tensor (presumably the data)
                x_val = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Handle the normal case where batch is a tuple of (data, target)
                x_val, _ = batch
            else:
                raise ValueError(f"Unexpected batch structure: {batch}")
        # reshape data to correct shape
        x_val = x_val.view(-1, 91)
        optimiser.zero_grad()

        output, mu, logvar = model(x_val)
        loss = loss_function(output, x_val, mu, logvar, l1, l2, model=model)
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / 50
    print(f"Average validation loss: {avg_val_loss}")

    # Return the average loss over the dataset as the objective to minimize
    return avg_val_loss


# Ensure that the loss function is correctly implemented
def loss_function(x_hat, x, mu, logvar, l1, l2, model):

    # make sure x_hat is a tensor
    x_hat = torch.tensor(x_hat, dtype=torch.float32)
    # Reconstruction loss
    recon_loss = nn.functional.cross_entropy(x_hat, x)
    # KL divergence
    epsilon = 1e-10
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar.exp() + epsilon))
    #     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # add l1 and l2 regularization
    l1_reg = torch.tensor(0.)
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
        l2_reg += torch.norm(param, 2)

    loss = recon_loss + kld + l1 * l1_reg + l2 * l2_reg
    return loss


def constrained_objective_function(particles, *args, **kwargs):
    # Adjust each particle
    for i in range(particles.shape[0]):
        particles[i, :] = enforce_hyperparameter_constraints(particles[i, :])

    # Evaluate the objective function for all particles
    fitness = np.apply_along_axis(objective_function, 1, particles)
    return fitness


# Setting up the PSO
options = {'c1': 2, 'c2': 2, 'w':0.9}  # PSO hyperparameters: cognitive constant,
# social constant, inertia coefficient
bounds = (np.array([1, 90, 50, 0, 0, 0]), np.array([50, 90, 500, 0.5, 1, 1]))  # Lower and upper bounds of the search space
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(constrained_objective_function, iters=50)

print("Optimized Cost:", cost)
print("Optimized Position:", pos)

# run vae with optimized hyperparameters
latent_dim = pos[0]
input_dim = pos[1]
hidden_dim = pos[2]
dropout = pos[3]
l1 = pos[4]
l2 = pos[5]

latent_dim = latent_dim.astype(int)
hidden_dim = hidden_dim.astype(int)
input_dim = input_dim.astype(int)
dropout = dropout.astype(float)
l1 = l1.astype(float)
l2 = l2.astype(float)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=latent_dim, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, activation=nn.Tanh)
# model.to(device)
optimiser = optim.Adam(model.parameters(), lr=1e-3)  # Adjust learning rate as needed

# Training Loop
model.train()
for epoch in range(100):
    print(f"Epoch {epoch + 1}")
    total_loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Check if the batch is just a tensor or a tuple/list of tensors
        if isinstance(batch, torch.Tensor):
            # Handle the case where the batch is just a single tensor (presumably the data)
            x = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            # Handle the normal case where batch is a tuple of (data, target)
            x, _ = batch
        else:
            raise ValueError(f"Unexpected batch structure: {batch}")
    # reshape data to correct shape
    x = x.view(-1, 91)

    optimiser.zero_grad()
    x_hat, mu, logvar = model(x)

    # Compute loss on validation set

    loss = loss_function(x_hat, x, mu, logvar, l1, l2, model=model)
    total_loss += loss.item()
    avg_loss = total_loss / 418
    print(f"Training Loss: {avg_loss}")

    # Backward pass
    loss.backward()
    optimiser.step()

model.eval()
total_val_loss = 0
with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        # Check if the batch is just a tensor or a tuple/list of tensors
        if isinstance(batch, torch.Tensor):
            # Handle the case where the batch is just a single tensor (presumably the data)
            x_val = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            # Handle the normal case where batch is a tuple of (data, target)
            x_val, _ = batch
        else:
            raise ValueError(f"Unexpected batch structure: {batch}")
    # reshape data to correct shape
    x_val = x_val.view(-1, 91)

    optimiser.zero_grad()

    output, mu, logvar = model(x_val)
    loss = loss_function(output, x_val, mu, logvar, l1, l2, model=model)
    total_val_loss += loss.item()
    avg_val_loss = total_val_loss / 50
    print(f"Average validation loss: {avg_val_loss}")

# generate latent space
z = torch.randn(500, latent_dim)
x_hat = model.decoder(z)
x_hat = x_hat.detach().numpy()

data = pd.read_csv('data/mci_preprocessed_wo_csf_vae.csv')
# save latent space
latent_space = pd.DataFrame(x_hat, columns=data.columns)
latent_space.to_csv('data/generated_cn_data2.csv', index=False)



# Training Loss: -0.28574844068317323
# Average validation loss: -0.5590079116821289
# time:2m:59



