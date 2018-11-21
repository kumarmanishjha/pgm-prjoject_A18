# *****************
# BASED ON Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at
# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
# Also inspired from https://github.com/pytorch/examples/blob/master/dcgan/main.py
# ******************

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from other import *
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import all the data
# Train shape (500, 2)
data_train_hmk = np.loadtxt('hwk3data/EMGaussian.train')
# Test shape also (500, 2)
data_test_hmk = np.loadtxt('hwk3data/EMGaussian.test')
data_total_hmk = np.vstack((data_train_hmk, data_test_hmk))

# CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Standard slope of the leak of LeakyReLU is 0.01, 0.2 recommended by DCGAN paper
# but 0.2 seems to work very badly on our 2d data sets
slope_leaky_relu = 0.01
n_epochs = 30000
# 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
d_steps = 1
g_steps = 1


using_synthetic_data = False
if using_synthetic_data:
    # the data is generated of the circumference
    # of a circle with some noise added
    real_data_generator = generator_circle_data(device, (2, 2))
    minibatch_size = 100
else:
    real_data_generator = homework_data_generator(data_total_hmk, device)
    minibatch_size = 100


class Generator(nn.Module):
    def __init__(self, dim_input, hidden_size, output_size):
        super(Generator, self).__init__()
        # fc for fully connected layer
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # From https://github.com/soumith/ganhacks,
        # use LeakyReLU to avoid sparse gradients
        x = F.leaky_relu(self.fc1(x), negative_slope=slope_leaky_relu)
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


dim_gen_input = real_data_generator(1).shape[1]
dim_gen_output = dim_gen_input
# number of neurons in hidden layer for generator
g_hidden_size = 50

G = Generator(dim_gen_input, g_hidden_size, dim_gen_output).to(device)
# We feed in the generator a noise input of a multivariate Normal(mu, sigma)
# Maybe test also uniform noise sampler ? (already in other.py)
g_noise_input_generator = gaussian_noise_sampler(0, 1, device)


class Discriminator(nn.Module):
    def __init__(self, dim_input, hidden_size, output_size):
        super(Discriminator, self).__init__()
        # fc for fully connected layer
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # From https://github.com/soumith/ganhacks,
        # use LeakyReLU to avoid sparse gradients
        x = F.leaky_relu(self.fc1(x), negative_slope=slope_leaky_relu)
        x = F.leaky_relu(self.fc2(x), negative_slope=slope_leaky_relu)
        # Training a standard binary classifier with a
        # sigmoid output (--> using Binary Cross Entropy Loss)
        return torch.sigmoid(self.fc3(x))


# Same as dim_generator_input
# --> the dimension of the data we try to generate
dim_dis_input = dim_gen_input
# number of neurons in hidden layer
d_hidden_size = 50

# Single dimension for 'real' vs. 'fake'
# We can change this if we want to use gans that also classify
d_output_size = 1

D = Discriminator(dim_dis_input, d_hidden_size, d_output_size).to(device)


# Learning rates recommended by the DCGAN paper for Adam
d_learning_rate = 2e-4
g_learning_rate = 2e-4
# From https://github.com/soumith/ganhacks, use Adam Optimizer
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
# use binary cross entropy loss function
loss_fn = nn.BCELoss()

arr_gradient_D = []
arr_gradient_G = []
arr_means = []
arr_std_dev = []
# set to true or false to plot the mean and std_dev
plot_means_and_var = False

real_label = 1
fake_label = 0


def train():
    for epoch in range(n_epochs):

        for d_index in range(d_steps):

            # (1) Update Discriminator : maximize log(D(x)) + log(1 - D(G(z)))

            # Reset gradients
            D.zero_grad()

            # on real samples
            d_real_data = real_data_generator(minibatch_size)
            d_real_decision = D(d_real_data)

            # To compute BCE loss
            label = torch.full((minibatch_size, 1), real_label, device=device)
            # D(x) should be 1 for real samples
            d_real_loss = loss_fn(d_real_decision, label)
            # compute/store gradients, but don't change params
            d_real_loss.backward()

            # on fake samples
            d_gen_input = g_noise_input_generator(minibatch_size, dim_gen_input)
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            # D(x) now wants to be as close as possible as fake label value
            label.fill_(fake_label)
            d_fake_loss = loss_fn(d_fake_decision, label)
            d_fake_loss.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

            # Compute the norm of gradient
            total_norm = 0
            for p in D.parameters():
                norm = p.grad.data.norm(2)
                total_norm += norm.item() ** 2
            total_norm = np.sqrt(total_norm)
            arr_gradient_D.append(total_norm)

        for g_index in range(g_steps):

            # (2) Update G network: maximize log(D(G(z)))
            G.zero_grad()

            gen_input = g_noise_input_generator(minibatch_size, dim_gen_input)
            gen_output = G(gen_input)
            d_fake_decision = D(gen_output)

            # The generator tries to get D(G(z)) near 1
            label = torch.full((minibatch_size, 1), real_label, device=device)
            g_loss = loss_fn(d_fake_decision,  label)

            g_loss.backward()
            g_optimizer.step()

            # Compute norm of gradient
            total_norm = 0
            for p in G.parameters():
                norm = p.grad.data.norm(2)
                total_norm += norm.item() ** 2
            total_norm = np.sqrt(total_norm)
            arr_gradient_G.append(total_norm)

        if epoch % 500 == 0:
            # Store mean/std_dev
            if plot_means_and_var:
                # Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                np_gen_output = gen_output.detach().numpy()
                mean = np.mean(np_gen_output)
                std_dev = np.std(np_gen_output)
                arr_means.append(mean)
                arr_std_dev.append(std_dev)
                print('mean = ', mean, ' std_dev = ', std_dev)

            print("%s: D_real_loss = %.4f; D_fake_loss = %.4f; G_loss = %.4f"
                  % (epoch, d_real_loss.item(), d_fake_loss.item(), g_loss.item()))


train()

# Process is over, let's plot some points of the generator
#  and their moments (mean, variance, etc..)
n_samples_plot = 500

g_noise_input = g_noise_input_generator(n_samples_plot, dim_gen_input)
g_noise_output = G(g_noise_input)

# Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
np_data_plot = g_noise_output.detach().numpy()

# Estimating 1d densities
if dim_gen_input == 1:
    mean_samples = np.mean(np_data_plot)
    std_dev_samples = np.std(np_data_plot)
    plt.hist(np_data_plot, bins=50)
    plt.title('Samples generated by the Generator with mean = %.2f, std_dev = %.2f'
              % (float(mean_samples), float(std_dev_samples)))
    plt.show()

elif dim_gen_input == 2:
    # Plot the real generating process
    data_toy = real_data_generator(500)

    plt.scatter(data_toy[:, 0], data_toy[:, 1], color='magenta', label='true distribution')

    # Plot fake data
    plt.scatter(np_data_plot[:, 0], np_data_plot[:, 1], color='blue', label='generated distribution')

    plt.legend()
    plt.show()


# Plot the gradient norms
plot_var_in_fct_of_epochs(arr_gradient_G, 'Generator', 'norm of gradient')
plot_var_in_fct_of_epochs(arr_gradient_D, 'Discriminator',  'norm of gradient')

# Plot mean/std_dev
if plot_means_and_var:
    plot_var_in_fct_of_epochs(arr_means, 'Sample means', 'mean')
    plot_var_in_fct_of_epochs(arr_std_dev, 'Sample std deviation',  'std_dev')