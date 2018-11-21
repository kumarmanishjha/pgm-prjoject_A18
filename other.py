import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb


def generate_1d_gaussian_data(mu, sigma, device):
    """ return random samples from 1d gaussian """
    # draw 1 x n samples
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)), device=device)


def homework_data_generator(data_total_hmk, device):
    """ take a mini batch of size n
    from the homework 3 data set """
    index = np.arange(len(data_total_hmk))
    np.random.shuffle(index)
    data_total_hmk = data_total_hmk[index]
    return lambda n: torch.Tensor(data_total_hmk[:n], device=device)


def generate_circle_points(n, center, device):
    """ return random samples from the circumference of a circle
        with added noise """
    # draw n thetas (angle)
    thetas = torch.rand(n) * 2 * np.pi
    x = np.cos(thetas) + center[0]
    y = np.sin(thetas) + center[1]

    # add random noise
    x += 0.25 * torch.rand(n)
    y += 0.25 * torch.rand(n)
    # reshape
    x = x.view(n, 1)
    y = y.view(n, 1)

    # put x in col[0] and y in col[1]
    return torch.cat((x, y), 1).to(device)


def generator_circle_data(device, center=(2, 2)):
    """ return random samples from the circumference of a circle
    with added noise """
    return lambda n: generate_circle_points(n, center, device)


def gaussian_noise_sampler(mu, sigma, device):
    """ Gaussian input to the generator"""
    return lambda m, n: torch.Tensor(np.random.normal(mu, sigma, (m, n)), device=device)


def uniform_noise_sampler(device):
    """ Uniform input to the generator of a m x n tensor"""
    return lambda m, n: torch.rand(m, n, device=device)


def plot_var_in_fct_of_epochs(norm_grads, title, y_label):
    plt.plot(np.arange(len(norm_grads)), norm_grads)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
