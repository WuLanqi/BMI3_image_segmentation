# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:15:48 2020

@author: sissy
"""

from __future__ import print_function

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#from keras.layers import Dropout
import matplotlib.pyplot as plt

import argparse
from keras.callbacks import *
import sys
#import theano
#import theano.tensor as T
from keras import initializers
from keras.layers import BatchNormalization
import copy

K.common.set_image_dim_ordering('th')  # Theano dimension ordering in this code

'''
    DEFAULT CONFIGURATIONS
'''
def get_options():

    parser = argparse.ArgumentParser(description='UNET for Lung Nodule Detection')

    parser.add_argument('--out_dir', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/output_final/',
                        dest="out_dir", type=str)
    parser.add_argument('--prediction_fig', action="store", default='/public/workspace/3180111438bit/bmi3_project/Luna2016/prediction_fig_test/',
                        dest="prediction_fig", type=str)
    

    parser.add_argument('--epochs', action="store", default=20, dest="epochs", type=int)

    parser.add_argument('--batch_size', action="store", default=1, dest="batch_size", type=int)

    parser.add_argument('--lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('--load_weights', action="store", default=False, dest="load_weights", type=bool)
    parser.add_argument('--filter_width', action="store", default=3, dest="filter_width",type=int)
    parser.add_argument('--start_filters', action="store", default=32, dest="start_filters",type=int)
    #parser.add_argument('--stride', action="store", default=3, dest="stride",type=int)
    parser.add_argument('--model_file', action="store", default="", dest="model_file",type=str) #TODO
    parser.add_argument('--save_prefix', action="store", default="model_",
                        dest="save_prefix",type=str)
    opts = parser.parse_args(sys.argv[1:])    


    return opts


#Dice coeffecient
smooth = 1.
def dice_coef(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)




def unet_model(options):
    input_shape = (1, 512, 512)
    inputs=Input(input_shape)
    filters=options.start_filters
    kernel_size=options.filter_width
    conv1 = Conv2D(filters,kernel_size,padding='same',activation='relu')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(filters,kernel_size,padding='same',activation='relu')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(filters*2,kernel_size,padding='same',activation='relu')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(filters*2,kernel_size,padding='same',activation='relu')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(filters*4,kernel_size,padding='same',activation='relu')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Conv2D(filters*4,kernel_size,padding='same',activation='relu')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(filters*8,kernel_size,padding='same',activation='relu')(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Conv2D(filters*8,kernel_size,padding='same',activation='relu')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

    conv5 = Conv2D(filters*16,kernel_size,padding='same',activation='relu')(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Conv2D(filters*16,kernel_size,padding='same',activation='relu')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv5)

    #bottleneck
    conv6 = Conv2D(filters*32,kernel_size,padding='same',activation='relu')(pool5)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(filters*32,kernel_size,padding='same',activation='relu')(conv6)


    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=1)
    conv7 = Dropout(0.3)(up7)
    conv7 = Conv2D(filters*16,kernel_size,padding='same',activation='relu')(conv7)
    conv7 = Conv2D(filters*16,kernel_size,padding='same',activation='relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=1)
    conv8 = Dropout(0.3)(up8)
    conv8 = Conv2D(filters*8,kernel_size,padding='same',activation='relu')(conv8)
    #conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(filters*8,kernel_size,padding='same',activation='relu')(conv8)
    #conv8 = BatchNormalization(axis=1)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=1)
    conv9 = Dropout(0.3)(up9)
    conv9 = Conv2D(filters*4,kernel_size,padding='same',activation='relu')(conv9)
    #conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(filters*4,kernel_size,padding='same',activation='relu')(conv9)
    #conv9 = BatchNormalization(axis=1)(conv9)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=1)
    conv10 = Dropout(0.3)(up10)
    conv10 = Conv2D(filters*2,kernel_size,padding='same',activation='relu')(conv10)
    #conv10 = BatchNormalization(axis=1)(conv10)
    conv10 = Conv2D(filters*2,kernel_size,padding='same',activation='relu')(conv10)
    #conv10 = BatchNormalization(axis=1)(conv10)

    up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=1)
    conv11 = Dropout(0.3)(up11)
    conv11 = Conv2D(filters,kernel_size,padding='same',activation='relu')(conv11)
    #conv11 = BatchNormalization(axis=1)(conv11)
    conv11 = Conv2D(filters,kernel_size,padding='same',activation='relu')(conv11)
    #conv11 = BatchNormalization(axis=1)(conv11)

    conv12 = Conv2D(1, (1, 1),padding='same',activation='sigmoid')(conv11)

    #optimizer: Adam (best choice)
    model = Model(input=inputs, output=conv12)
    model.summary()
    model.compile(optimizer = Adam(lr=options.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])
    return model



print ("Loading the options ....")
options = get_options()
print ("epochs: %d"%options.epochs)
print ("batch_size: %d"%options.batch_size)
print ("filter_width: %d"%options.filter_width)
print ("start_filters: %d"%options.start_filters)
#print ("stride: %d"%options.stride)
print ("learning rate: %f"%options.lr)
sys.stdout.flush()

print('-'*30)
print('Loading and preprocessing train and test data...')
print('-'*30)
imgs_train = np.load(options.out_dir+"trainImages.npy").astype(np.float32)
imgs_mask_train = np.load(options.out_dir+"trainMasks.npy").astype(np.float32)
# Renormalizing the masks
imgs_mask_train[imgs_mask_train > 0.] = 1.0

# Now the Test Data
imgs_test = np.load(options.out_dir+"testImages.npy").astype(np.float32)
imgs_mask_test_true = np.load(options.out_dir+"testMasks.npy").astype(np.float32)
# Renormalizing the test masks
imgs_mask_test_true[imgs_mask_test_true > 0] = 1.0


mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean  # images should already be standardized, but just in case
imgs_train /= std

print('-'*30)
print('Creating and compiling model...')
print('-'*30)
model = unet_model(options)
model_checkpoint = ModelCheckpoint('model_test.hdf5', monitor='loss', verbose = 1, save_best_only=True)


print('-'*30)
print('Fitting model...')
print('-'*30)
history = model.fit(x=imgs_train, y=imgs_mask_train, batch_size=options.batch_size, nb_epoch=options.epochs, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data = (imgs_test, imgs_mask_test_true))


plt.plot(history.history['dice_coef'],color='b')
plt.plot(history.history['val_dice_coef'], color='g')
plt.xlabel("Epoch")
plt.ylabel("Dice Coefficient")
plt.legend(["Train", "Validation"])
plt.savefig(options.out_dir + 'line_plot_test.jpg', bbox_inches='tight', dpi=150)

print('-'*30)
print('Evaluating model...')
print('-'*30)
model.evaluate(imgs_test,imgs_mask_test_true, batch_size=4)

############################## Predict ##########################
print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
num_test=imgs_test.shape[0]
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
for i in range(num_test):
    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
np.save('masksTestPredicted_test.npy', imgs_mask_test)
mean = 0.0
for i in range(num_test):
    mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
mean/=num_test
print("Mean Dice Coeff : ",mean)


pre_fig_dir = options.prediction_fig
for index in range(200):
    plt.figure(1)
    plt.subplot(231)
    plt.imshow(imgs_mask_test[index,0], cmap="gray")
    plt.subplot(232)
    plt.imshow(imgs_mask_test_true[index,0],cmap="gray")
    plt.subplot(234)
    plt.imshow(imgs_test[index,0], cmap="gray")
    plt.show()
    plt.savefig(pre_fig_dir+"prediction_"+str(index)+".png")
