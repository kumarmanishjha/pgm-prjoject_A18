# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:02:23 2018

@author: Gabriel Hsu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import pickle
from skimage.io import imsave
import itertools 

from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Conv2DTranspose,  BatchNormalization, Reshape, UpSampling2D, Concatenate, Conv2D, AveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras import metrics

##%%
##MNIST
#num_classes = 10
#(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#y_train = to_categorical(y_train_, num_classes)
#y_test = to_categorical(y_test_, num_classes)

#%% CIFAR-10


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = 10

img_rows, img_cols = 32, 32
channels = 3
input_shape = (img_rows, img_cols, channels)

def load_pickle(f):
        return  pickle.load(f, encoding='latin1')

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'D:/ML_project/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    x_val = X_val.astype('float32')
    x_train /= 255
    x_test /= 255
    x_val /= 255
    

    return x_train, y_train, x_val, y_val, x_test, y_test


# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()


print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

y_train = to_categorical(y_train, num_classes)
y_val =  to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

def to_image(data, row, col, channel):
    size = data.shape[0]
    tp = np.zeros((size, row, col, channel))
    for i in range(size):
        tp[i,:,:,:] = np.transpose(np.reshape(data[i,:],(3, 32,32)), (1,2,0))
    return tp
        
x_train = to_image(x_train, img_rows, img_cols, channels)
x_val = to_image(x_val, img_rows, img_cols, channels)
x_test = to_image(x_test, img_rows, img_cols, channels)

#%% LOSS

class KLLossLayer(Layer):
    __name__ = 'kl_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(KLLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_avg, z_log_var):
        kl_loss = -0.5 * K.mean(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var))
        return kl_loss

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        loss = self.lossfun(z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return z_avg
    
class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_true, y_pred):
        loss_x = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)


        return loss_x 

    def call(self, inputs):
        y_true = inputs[0]
        y_pred = inputs[1]
        loss = self.lossfun(y_true, y_pred)
        self.add_loss(loss, inputs=inputs)

        return y_true
    
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

#%%
#The number of sample from z 
m = 128

#dimension of the latent variable (can decide by ourselves)
latent_dim = 100 

#epochs
n_epoch = 2
#Q(z|X) -- encoder
img = Input(shape=input_shape)
label = Input(shape=(num_classes,))
c = Reshape((1, 1, num_classes))(label)
c = UpSampling2D(size=(img_rows, img_cols))(c)
en = Concatenate(axis=-1)([img, c])

en = Conv2D(32, (3, 3), padding='same', activation='relu')(en)
en = Conv2D(32, (3, 3), padding='same', activation='relu')(en)
en = BatchNormalization()(en)
en = AveragePooling2D(pool_size=(2, 2))(en)
en = Dropout(0.2)(en)

en = Conv2D(64, (3, 3), padding='same', activation='relu')(en)
en = Conv2D(64, (3, 3), padding='same', activation='relu')(en)
en = BatchNormalization()(en)
BatchNormalization()
en = AveragePooling2D(pool_size=(2, 2))(en)
en = Dropout(0.2)(en)

en = Conv2D(128, (3, 3), padding='same', activation='relu')(en)
en = Conv2D(128, (3, 3), padding='same', activation='relu')(en)
en = BatchNormalization()(en)
BatchNormalization()
en = AveragePooling2D(pool_size=(2, 2))(en)
en = Dropout(0.2)(en)

en = Flatten()(en)

#calculate the mu and sigmas 
mu = Dense(latent_dim, activation='linear')(en)
log_sigma = Dense(latent_dim, activation='linear')(en)

def sample_z(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#Sample z ~ Q(z|X)
z = Lambda(sample_z, output_shape=(latent_dim,))([mu, log_sigma])


#P(X|z) -- decoder
#decoder
#Build Decoder
#dec = Input(shape=(latent_dim,))
dec_in = Concatenate(axis=-1)([z, label])
de = Dense(4*4*128, activation='relu')(dec_in)
de = Reshape((4,4,128))(de)

de = Conv2DTranspose(256, (1,1), strides=(2, 2))(de)
de = Conv2D(128, (3, 3), padding='same', activation='relu')(de)
de = Conv2D(128, (3, 3), padding='same', activation='relu')(de)
de = BatchNormalization()(de)

de = Conv2DTranspose(128, (2,2), strides=(2, 2))(de)
de = Conv2D(64, (3, 3), padding='same', activation='relu')(de)
de = Conv2D(64, (3, 3), padding='same', activation='relu')(de)
de = BatchNormalization()(de)

de = Conv2DTranspose(64, (2,2), strides=(2, 2))(de)
de = Conv2D(32, (3, 3), padding='same', activation='relu')(de)
de = Conv2D(32, (3, 3), padding='same', activation='relu')(de)
de = BatchNormalization()(de)

h_decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(de)


#h_p = decoder_hidden(z)
#outputs = decoder_out(h_p)

#Overall VAE model


#encoder model
#encoder = Model(img, [mu, log_sigma, z])
#print ("ENCODER")
#encoder.summary()

#decoder model
#decoder = Model(dec,h_decoded)
#print ("DECODER")
#decoder.summary()

#outputs = decoder(z)
cvae = Model([img, label], h_decoded)

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
xent_loss = img_rows * img_cols * channels * metrics.binary_crossentropy(
    K.flatten(img),
    K.flatten(h_decoded))
kl_loss = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
cvae_loss = K.mean(xent_loss + kl_loss)
cvae.add_loss(cvae_loss)
cvae.compile(optimizer = 'adam')
cvae.summary()
cvae.fit([x_train,y_train], batch_size=m, epochs = n_epoch)
#%%
encoder = Model([img, label], [mu, log_sigma, z])

decoder_input = Input(shape=(latent_dim,))
d = Concatenate(axis=-1)([decoder_input, label])
d = Dense(4*4*128, activation='relu')(decoder_input)
d = Reshape((4,4,128))(d)
d = BatchNormalization()(d)

d = Conv2DTranspose(1024, (1,1), strides=(2, 2))(d)
d = Conv2D(512, (3, 3), padding='same', activation='relu')(d)
d = Conv2D(512, (3, 3), padding='same', activation='relu')(d)
d = BatchNormalization()(d)

d = Conv2DTranspose(512, (2,2), strides=(2, 2))(d)
d = Conv2D(256, (3, 3), padding='same', activation='relu')(d)
d = Conv2D(256, (3, 3), padding='same', activation='relu')(d)
d = BatchNormalization()(d)

d = Conv2DTranspose(256, (2,2), strides=(2, 2))(d)
d = Conv2D(128, (3, 3), padding='same', activation='relu')(d)
d = Conv2D(128, (3, 3), padding='same', activation='relu')(d)
d = BatchNormalization()(d)

decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)
decoder = Model(decoder_input, decoder_output)



#%% training

def save_batch_result(batch_data, path, epoch):
    batch_size = batch_data.shape[0]
    for i in range(batch_size):
        f_name = str(epoch) + '_' + str(i) + '.png'
        img_sav = batch_data[i]
        imsave(os.path.join(path,f_name), img_sav)

def show_result(batch_data, path, epoch, show):
      size_figure_grid = 5
      fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
      for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
          ax[i, j].get_xaxis().set_visible(False)
          ax[i, j].get_yaxis().set_visible(False)

      for k in range(5*5):
          i = k // 5
          j = k % 5
          ax[i, j].cla()
          ax[i, j].imshow(batch_data[k])

      label = 'Epoch {0}'.format(epoch)
      fig.text(0.5, 0.04, label, ha='center')
      plt.savefig(os.path.join(path, str(epoch)))

      if show:
          plt.show()
      else:
          plt.close()
          
#def train_on_batch(x_batch, epoch):
#    x_r, c = x_batch
#    x_dummy = np.zeros(x_r.shape, dtype='float32')
##    z_dummy = np.zeros(z.shape, dtype='float32')
#
#    # Train autoencoder
#    loss = cvae.train_on_batch([x_r, c], x_dummy)
#    
#
#    return loss
#
#
##%% Training 
#n_epoch = 20
#
## results save folder
#if not os.path.isdir('CVAE_Random_results'):
#    os.mkdir('CVAE_Random_results')
#if not os.path.isdir('CVAE_Fixed_results'):
#    os.mkdir('CVAE_Fixed_results')
#if not os.path.isdir('CVAE_Results'):
#    os.mkdir('CVAE_Results')    
#
#
#size = x_train.shape[0]
#loss = 0
#
#for epoch in range(n_epoch):
#    print ("epochs: ", epoch)
#    print ("Losses", loss)
#    for i in range(int(size/m)):
#        idx = np.random.randint(0, x_train.shape[0], m)
#        imgs = x_train[idx]
#        labels = y_train[idx]
#        loss = train_on_batch([imgs, labels], epoch)
#    #save input image
#    save_batch_result(imgs[0:50], 'CVAE_Fixed_results', epoch)
#    #save generated image
#    f_latent = encoder.predict([imgs[0:50], labels[0:50]])
#    f_image = decoder.predict([f_latent, labels[0:50]])
#    save_batch_result(f_image, 'CVAE_Random_results', epoch)
#    show_result(f_image, 'CVAE_Results', epoch, True)


#%%

#randomly sample one sample from N(0, I)
mean = np.zeros((latent_dim))
cov = np.identity(latent_dim)

sample = np.random.multivariate_normal(mean, cov, 1000)

pred = decoder.predict(sample, batch_size = m)

a = (pred[0,:]*255).astype(np.uint8)


plt.imshow(a, cmap = 'gray')




