# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:02:23 2018

@author: Gabriel Hsu
"""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

#MNIST
num_classes = 10
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)

m = 50
n_x = x_train.shape[1]
n_y = y_train.shape[1]

n_epoch = 20

original_dim = 784
intermediate_dim = 512
latent_dim= 2

#Use image and label as input together

# Q(z|X,y) -- encoder
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# calculate the mu and sigmas 
mu = Dense(latent_dim)(h)
log_sigma = Dense(latent_dim)(h)

y = Input(shape=(num_classes,))
yh = Dense(latent_dim)(y)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, latent_dim))
    return mu + K.exp(log_sigma / 2) * eps

# Sample z ~ Q(z|X,y)
z = Lambda(sample_z)([mu, log_sigma])

#decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

#whole model
vae = Model([x, y], [x_decoded_mean, yh])

#define loss
reconstruction_loss = binary_crossentropy(x, x_decoded_mean)
reconstruction_loss *= 784
kl_loss = 1 + log_sigma - K.square(mu-yh) - K.exp(log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer = 'adam')
vae.summary()
vae.fit([x_train, y_train], batch_size=m, epochs = 50)

#%% Generate image from here (the same way as VAE)


