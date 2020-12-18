from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

import argparse
import sys
from keras.layers import BatchNormalization

def get_options():   
    parser = argparse.ArgumentParser(description='Step 3.1: plot prediction')
    parser.add_argument('--working_dir', action="store", default='/public/workspace/3180111428bit/bmi3_project/Luna2016/',
                        dest="working_dir", type=str)
    parser.add_argument('--out_dir', action="store", default='/public/workspace/3180111428bit/bmi3_project/Luna2016/output_final/',
                        dest="out_dir", type=str)
    parser.add_argument('--pre_fig_dir', action="store", default='/public/workspace/3180111428bit/bmi3_project/Luna2016/3_1_plots/',
                        dest="pre_fig_dir", type=str)
    parser.add_argument('--lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('--start_filters', action="store", default=32, dest="start_filters",type=int)
    parser.add_argument('--filter_width', action="store", default=3, dest="filter_width",type=int)
    opts = parser.parse_args(sys.argv[1:]) # allow multiple parameter
    return opts

options = get_options()

K.common.set_image_dim_ordering('th')  # Theano dimension ordering in this code

#Dice coeffecient
smooth = 1. # avoid dividing by zero
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

def compare(fig1, fig2):
    return fig1 == fig2

def accuracy(fig1, fig2):
    size = (fig1.shape[0])*(fig2.shape[1])
    acc = np.sum(compare(fig1,fig2)) / size
    return acc

def fault_alarm(dice):
    if dice < 0.5:
        message = '!!!ALARM!!! \n The prediction may not be true, please judge it by yourself! \n !!!ALARM!!!'
    else:
        message = 'Good Prediction'
    return message
    

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
    conv6 = BatchNormalization(axis=1)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=1)
    conv7 = Conv2D(filters*16,kernel_size,padding='same',activation='relu')(up7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(filters*16,kernel_size,padding='same',activation='relu')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=1)
    conv8 = Conv2D(filters*8,kernel_size,padding='same',activation='relu')(up8)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(filters*8,kernel_size,padding='same',activation='relu')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=1)
    conv9 = Conv2D(filters*4,kernel_size,padding='same',activation='relu')(up9)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(filters*4,kernel_size,padding='same',activation='relu')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=1)
    conv10 = Conv2D(filters*2,kernel_size,padding='same',activation='relu')(up10)
    conv10 = BatchNormalization(axis=1)(conv10)
    conv10 = Conv2D(filters*2,kernel_size,padding='same',activation='relu')(conv10)
    conv10 = BatchNormalization(axis=1)(conv10)

    up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=1)
    conv11 = Conv2D(filters,kernel_size,padding='same',activation='relu')(up11)
    conv11 = BatchNormalization(axis=1)(conv11)
    conv11 = Conv2D(filters,kernel_size,padding='same',activation='relu')(conv11)
    conv11 = BatchNormalization(axis=1)(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    #optimizer: Adam (best choice)
    model = Model(input=inputs, output=conv12)
    model.summary()
    model.compile(optimizer = Adam(lr=options.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])
    return model


working_dir = options.working_dir
model = unet_model(options)
model.load_weights(working_dir + "model_3_unet_epoch12.hdf5")

out_dir = options.out_dir
# Load test data
imgs_test = np.load(out_dir+"testImages.npy").astype(np.float32)
imgs_mask_test_true = np.load(out_dir+"testMasks.npy").astype(np.float32)
# Renormalizing the test masks
imgs_mask_test_true[imgs_mask_test_true > 0] = 1.0

num_test=imgs_test.shape[0]
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
imgs_mask_test[imgs_mask_test > 0] = 1.0

###################### CAN CHANGE INDEX LIST #########################
index_list = [47, 53, 111, 159, 181, 185, 187, 194, 195]
######################################################################

pre_fig_dir = options.pre_fig_dir
for index in index_list:
    print("Predicting on: "+str(index))
    imgs_mask_test[index] = model.predict([imgs_test[index:index+1]], verbose=0)[0]
    imgs_mask_test[imgs_mask_test < 0.6] = 0.0
    imgs_mask_test[imgs_mask_test > 0.6] = 1.0
    dc=dice_coef_np(imgs_mask_test_true[index,0], imgs_mask_test[index,0])
    acc = accuracy(imgs_mask_test[index,0], imgs_mask_test_true[index,0])
    mes = fault_alarm(dc)
    fig, (axs1,axs2,axs3) = plt.subplots(1,3, sharey=True)
    fig.suptitle(str(mes)+" \n "+"Pixel accuracy: "+str(acc)+"\n Dice coefficient: "+str(dc))
    axs1.imshow(imgs_mask_test[index,0], cmap="gray")
    axs1.set_title("Predicted")
    axs2.imshow(imgs_mask_test_true[index,0],cmap="gray")
    axs2.set_title("Ground Truth")
    axs3.imshow(imgs_test[index,0], cmap="gray")
    axs3.set_title("Image")
    fig.savefig(pre_fig_dir+"prediction_"+str(index)+".png")
