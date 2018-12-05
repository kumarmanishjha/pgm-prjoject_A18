# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:58:33 2018

@author: Gabriel Hsu
"""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.callbacks import LearningRateScheduler


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#%%

#The number of sample from z 
m = 128

#dimension of the latent variable (can decide by ourselves)
n_z = 2 

#epochs
n_epoch = 50
#Q(z|X) -- encoder
inputs = Input(shape=(784,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z)(h_q)
log_sigma = Dense(n_z)(h_q)

def sample_z(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])


#P(X|z) -- decoder
decoder_hidden = Dense(512, activation = 'relu')
decoder_out = Dense(784, activation = 'sigmoid')

#h_p = decoder_hidden(z)
#outputs = decoder_out(h_p)

#Overall VAE model


#encoder model
encoder = Model(inputs, [mu, log_sigma, z])

#decoder model
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)

#def vae_loss(y_true, y_pred):
#    """loss = reconstruction loss + KL loss for each data batch"""
#    
#    # E[log P(X|z)]
#    recon = K.sum(binary_crossentropy(y_pred, y_true))
#    
#    # D_KL(Q(z|x) || P(z|X))
#    kl = -0.5 * K.sum( 1. + log_sigma - K.exp(log_sigma) - K.square(mu), axis = -1)
#    
#    return K.mean(recon + kl)


#define loss
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + log_sigma - K.square(mu) - K.exp(log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer = 'adam')
vae.summary()
vae.fit(x_train, batch_size=m, epochs = n_epoch)

#%% Produce image by running Encoder and Decoder once and reconstruct

#randomly sample one sample from N(0, I)
mean = np.zeros((n_z))
cov = np.identity(n_z)

sample = np.random.multivariate_normal(mean, cov, 1000)

pred = decoder.predict(sample, batch_size = m)

#%%

plt.imshow((pred[500,:]*255).reshape(28, 28))




