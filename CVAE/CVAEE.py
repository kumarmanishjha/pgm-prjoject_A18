# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:34 2018

@author: Gabriel Hsu
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from skimage.io import imsave
import itertools 

from keras.metrics import binary_accuracy
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, MaxPooling2D, BatchNormalization, Lambda, Reshape, UpSampling2D, Conv2DTranspose, Concatenate, Conv2D, AveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.engine.topology import Layer
from keras.optimizers import Adam



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
    
class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_f, y_fake_p):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = K.binary_crossentropy(y_pos, y_real)
        loss_fake_f = K.binary_crossentropy(y_neg, y_fake_f)
        loss_fake_p = K.binary_crossentropy(y_neg, y_fake_p)
        return K.mean(loss_real + loss_fake_f + loss_fake_p)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        y_fake_p = inputs[2]
        loss = self.lossfun(y_real, y_fake_f, y_fake_p)
        self.add_loss(loss, inputs=inputs)

        return y_real


class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_pred):
        return K.mean(K.categorical_crossentropy(c_true, c_pred))

    def call(self, inputs):
        c_true = inputs[0]
        c_pred = inputs[1]
        loss = self.lossfun(c_true, c_pred)
        self.add_loss(loss, inputs=inputs)

        return c_true

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_r, x_f, f_D_x_f, f_D_x_r, f_C_x_r, f_C_x_f):
        loss_x = K.mean(K.square(x_r - x_f))
        loss_d = K.mean(K.square(f_D_x_r - f_D_x_f))
        loss_c = K.mean(K.square(f_C_x_r - f_C_x_f))

        return loss_x + loss_d + loss_c

    def call(self, inputs):
        x_r = inputs[0]
        x_f = inputs[1]
        f_D_x_r = inputs[2]
        f_D_x_f = inputs[3]
        f_C_x_r = inputs[4]
        f_C_x_f = inputs[5]
        loss = self.lossfun(x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f)
        self.add_loss(loss, inputs=inputs)

        return x_r

class FeatureMatchingLayer(Layer):
    __name__ = 'feature_matching_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(FeatureMatchingLayer, self).__init__(**kwargs)

    def lossfun(self, f1, f2):
        f1_avg = K.mean(f1, axis=0)
        f2_avg = K.mean(f2, axis=0)
        return 0.5 * K.mean(K.square(f1_avg - f2_avg))

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        loss = self.lossfun(f1, f2)
        self.add_loss(loss, inputs=inputs)

        return f1

#%%
#MNIST
#num_classes = 10
#(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
##x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
##x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#y_train = to_categorical(y_train_, num_classes)
#y_test = to_categorical(y_test_, num_classes)
#
#
#x_train = np.expand_dims(x_train , axis = -1)
#x_test = np.expand_dims(x_test, axis = -1)


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
#    mask = range(num_training, num_training + num_validation)
#    X_val = X_train[mask]
#    y_val = y_train[mask]
#    mask = range(num_training)
#    X_train = X_train[mask]
#    y_train = y_train[mask]
#    mask = range(num_test)
#    X_test = X_test[mask]
#    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
#    x_val = X_val.astype('float32')
    x_train /= 255
    x_test /= 255
#    x_val /= 255
    

    return x_train, y_train, x_test, y_test


# Invoke the above function to get our data.
x_train, y_train, x_test, y_test = get_CIFAR10_data()


print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
#print('Validation data shape: ', x_val.shape)
#print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

y_train = to_categorical(y_train, num_classes)
#y_val =  to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

def to_image(data, row, col, channel):
    size = data.shape[0]
    tp = np.zeros((size, row, col, channel))
    for i in range(size):
        tp[i,:,:,:] = np.transpose(np.reshape(data[i,:],(3, 32,32)), (1,2,0))
    return tp
        
x_train = to_image(x_train, img_rows, img_cols, channels)
#x_val = to_image(x_val, img_rows, img_cols, channels)
x_test = to_image(x_test, img_rows, img_cols, channels)



#%%
m = 128
n_x = x_train.shape[1]
n_y = y_train.shape[1]

n_epoch = 20

latent_dim= 100

#Use image and label as input together

#Build Encoder
img = Input(input_shape)
label = Input(shape=(num_classes,))

c = Reshape((1, 1, num_classes))(label)
c = UpSampling2D(size=(img_rows, img_cols))(c)
en = Concatenate(axis=-1)([img, c])

en = Conv2D(32, (3, 3), padding='same', activation='relu')(en)
en = BatchNormalization()(en)
en = MaxPooling2D(pool_size=(2, 2))(en)
en = Dropout(0.2)(en)

en = Conv2D(64, (3, 3), padding='same', activation='relu')(en)
en = BatchNormalization()(en)
en = MaxPooling2D(pool_size=(2, 2))(en)
en = Dropout(0.2)(en)

en = Conv2D(128, (3, 3), padding='same', activation='relu')(en)
en = BatchNormalization()(en)
en = MaxPooling2D(pool_size=(2, 2))(en)
en = Dropout(0.2)(en)

en = Flatten()(en)


#calculate the mu and sigmas 
mu = Dense(latent_dim)(en)
log_sigma = Dense(latent_dim)(en)


encoder = Model([img, label], mu)
print ("ENCODER")
encoder.summary()

#KL loss 
kl_loss = KLLossLayer()([mu, log_sigma])

#function of sample z randomly
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, latent_dim))
    return mu + K.exp(log_sigma / 2) * eps


#Build Decoder
dec = Input(shape=(latent_dim,))
dec_in = Concatenate(axis=-1)([dec, label])
de = Dense(4*4*128, activation='relu')(dec_in)
de = Reshape((4,4,128))(de)

de = Conv2DTranspose(256, (1,1), strides=(2, 2))(de)
de = Conv2D(128, (3, 3), padding='same', activation='relu')(de)
de = BatchNormalization()(de)

de = Conv2DTranspose(128, (2,2), strides=(2, 2))(de)
de = Conv2D(64, (3, 3), padding='same', activation='relu')(de)
de = BatchNormalization()(de)

de = Conv2DTranspose(64, (2,2), strides=(2, 2))(de)
de = Conv2D(32, (3, 3), padding='same', activation='relu')(de)
de = BatchNormalization()(de)

h_decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(de)
decoder = Model([dec, label], h_decoded)
print ("DECODER")
decoder.summary()

# Sample z 
z = Lambda(sample_z)([mu, log_sigma])
z_p = Input(shape=(latent_dim,))
#one fake data
x_f = decoder([z, label])
#one real data
x_p = decoder([z_p, label])

#discriminator 
dis_in = Input(shape=input_shape)
di = Conv2D(32, (3, 3), padding='same', activation='relu')(dis_in)
di = BatchNormalization()(di)
di = MaxPooling2D(pool_size=(2, 2))(di)
di = Dropout(0.2)(di)

di = Conv2D(64, (3, 3), padding='same', activation='relu')(di)
di = BatchNormalization()(di)
di = MaxPooling2D(pool_size=(2, 2))(di)
di = Dropout(0.2)(di)

di = Conv2D(128, (3, 3), padding='same', activation='relu')(di)
di = BatchNormalization()(di)
di = MaxPooling2D(pool_size=(2, 2))(di)
di = Dropout(0.2)(di)

di = Flatten()(di)
dis_out = Dense(1, activation='sigmoid')(di)

discriminator = Model(dis_in, [dis_out, di])
print ("DISCRIMINATOR")
discriminator.summary()

#real image input 
y_r , y_r_feature = discriminator(img)
#fake data input 
y_f , y_f_feature = discriminator(x_f)
#real data input
y_p , y_p_feature = discriminator(x_p)


#discriminator loss
d_loss = DiscriminatorLossLayer()([y_r, y_f, y_p])


#classificator
cl_in = Input(shape= input_shape)
cl = Conv2D(32, (3, 3), padding='same', activation='relu')(cl_in)
cl = BatchNormalization()(cl)
cl = MaxPooling2D(pool_size=(2, 2))(cl)
cl = Dropout(0.2)(cl)

cl = Conv2D(64, (3, 3), padding='same', activation='relu')(cl)
cl = BatchNormalization()(cl)
cl = MaxPooling2D(pool_size=(2, 2))(cl)
cl = Dropout(0.2)(cl)

cl = Conv2D(128, (3, 3), padding='same', activation='relu')(cl)
cl = BatchNormalization()(cl)
cl = MaxPooling2D(pool_size=(2, 2))(cl)
cl = Dropout(0.2)(cl)

cl = Flatten()(cl)
cl_out = Dense(num_classes, activation='softmax')(cl)
classificator = Model(cl_in, [cl_out, cl])
print ("CLASSIFICATOR")
classificator.summary()

#real image input 
c_r , c_r_feature = classificator(img)
#fake data input 
c_f , c_f_feature = classificator(x_f)
#real data input
c_p , c_p_feature = classificator(x_p)



#generate loss
g_loss = GeneratorLossLayer()([img, x_f, y_r_feature, y_f_feature, c_r_feature, c_f_feature])
gd_loss = FeatureMatchingLayer()([y_r_feature, y_p_feature])
gc_loss = FeatureMatchingLayer()([c_r_feature, c_p_feature])


#classification loss
c_loss = ClassifierLossLayer()([label, y_r])

#establish the trainer for each model
#set trainnable
def set_trainable(model, train):
    """
    Enable or disable training for the model
    args:
        model(?):
        train(?):
    """
    model.trainable = train
    for l in model.layers:
        l.trainable = train
        
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)


def discriminator_accuracy(x_r, x_f, x_p):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_r)
        x_neg = K.zeros_like(x_r)
        loss_r = K.mean(binary_accuracy(x_pos, x_r))
        loss_f = K.mean(binary_accuracy(x_neg, x_f))
        loss_p = K.mean(binary_accuracy(x_neg, x_p))
        return (1.0 / 3.0) * (loss_r + loss_p + loss_f)

    return accfun

def generator_accuracy(x_p, x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_p)
        loss_p = K.mean(binary_accuracy(x_pos, x_p))
        loss_f = K.mean(binary_accuracy(x_pos, x_f))
        return 0.5 * (loss_p + loss_f)

    return accfun

        
# Build classifier trainer
set_trainable(encoder, False)
set_trainable(decoder, False)
set_trainable(discriminator, False)
set_trainable(classificator, True)

cls_trainer = Model(inputs=[img, label],
                         outputs=[c_loss])
cls_trainer.compile(loss=[zero_loss],
                         optimizer=Adam(lr=2.0e-4, beta_1=0.5))

print ("CLASSIFICATOR TRAINER")
cls_trainer.summary()

# Build discriminator trainer
set_trainable(encoder, False)
set_trainable(decoder, False)
set_trainable(discriminator, True)
set_trainable(classificator, False)

dis_trainer = Model(inputs=[img, label, z_p],
                         outputs=[d_loss])
dis_trainer.compile(loss=[zero_loss],
                         optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                         metrics=[discriminator_accuracy(y_r, y_f, y_p)])

print ("DISCRIMINATOR　TRAINER")
dis_trainer.summary()

# Build generator trainer
set_trainable(encoder, False)
set_trainable(decoder, True)
set_trainable(discriminator, False)
set_trainable(classificator, False)

dec_trainer = Model(inputs=[img, label, z_p],
                         outputs=[g_loss, gd_loss, gc_loss])
dec_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
                         optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                         metrics=[generator_accuracy(y_p, y_f)])

# Build autoencoder
set_trainable(encoder, True)
set_trainable(decoder, False)
set_trainable(discriminator, False)
set_trainable(classificator, False)

enc_trainer = Model(inputs=[img, label, z_p],
                        outputs=[g_loss, kl_loss])
enc_trainer.compile(loss=[zero_loss, zero_loss],
                        optimizer=Adam(lr=2.0e-4, beta_1=0.5))

print ("ENCODER TRAINER")
enc_trainer.summary()

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
    


def train_on_batch(x_batch, epoch):
    x_r, c = x_batch
    
    batchsize = x_r.shape[0]
    z_p = np.random.normal(size=(batchsize, latent_dim)).astype('float32')

    x_dummy = np.zeros(x_r.shape, dtype='float32')
    c_dummy = np.zeros(c.shape, dtype='float32')
    z_dummy = np.zeros(z_p.shape, dtype='float32')
    y_dummy = np.zeros((batchsize, 1), dtype='float32')
    f_dummy = np.zeros((batchsize, 8192), dtype='float32')

    # Train autoencoder
    enc_trainer.train_on_batch([x_r, c, z_p], [x_dummy, z_dummy])

    # Train generator
    g_loss, _, _,  _, _, _, g_acc = dec_trainer.train_on_batch([x_r, c, z_p], [x_dummy, f_dummy, f_dummy])

    
    # Train classifier
    cls_trainer.train_on_batch([x_r, c], c_dummy)

    # Train discriminator
    d_loss, d_acc = dis_trainer.train_on_batch([x_r, c, z_p], y_dummy)

    loss = {
        'g_loss': g_loss,
        'd_loss': d_loss,
        'g_acc': g_acc,
        'd_acc': d_acc
    }
    return loss

def predict(z_samples):
    return decoder.predict(z_samples)

#%% Training 
n_epoch = 20

# results save folder
if not os.path.isdir('Random_results'):
    os.mkdir('Random_results')
if not os.path.isdir('Fixed_results'):
    os.mkdir('Fixed_results')
if not os.path.isdir('results'):
    os.mkdir('Results')



#%%%
size = x_train.shape[0]
loss = 0

for epoch in range(n_epoch):
    print ("epochs: ", epoch)
    print ("Losses", loss)
    for i in range(int(size/m)):
        idx = np.random.randint(0, x_train.shape[0], m)
        imgs = x_train[idx]
        labels = y_train[idx]
        loss = train_on_batch([imgs, labels], epoch)
    #save input image
    save_batch_result(imgs[0:50], 'Fixed_results', epoch)
    #save generated image
    f_latent = encoder.predict([imgs[0:50], labels[0:50]])
    f_image = decoder.predict([f_latent, labels[0:50]])
    save_batch_result(f_image, 'Random_results', epoch)
    show_result(f_image, 'Results', epoch, True)

#%% generate image
num_generated = 10

#the number you want to produce
digit = 9

ydd = np.ones(num_generated )*digit

sample =  np.random.normal(size=(num_generated , latent_dim)).astype('float32')
yd = to_categorical(ydd, num_classes)

pred = decoder.predict([sample, yd], batch_size = num_generated )

#%% show generated Image


for i in range(num_generated):
    img = pred[i,:].reshape(input_shape)
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    
    
