"""Libraries"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import cv2 as cv
import datetime
import matplotlib.pyplot as plt

"""Model Definition"""
def batchnorm_relu(inputs):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    #Convolutional Layer
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
    
    #Shortcut
    shortcut = Conv2D(num_filters, (1,1), padding='same', strides=strides)(inputs)
               
    #Addition ofthe convolutional layer and shortcut
    x = x + shortcut
    
    return x

def decoder_block(inputs, skip_features, num_filters):
    # = UpSampling2D((2, 2))(inputs)
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x

def dice_loss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

def loss_sum(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    o = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)

def build_resunet(img_height, img_width, channels):
    
    #Inputs
    inputs = Input((img_height, img_width, channels))
    inputs = Lambda(lambda x: x / 255)(inputs) #Normalize the pixels by dividing by 255
    
    #Encoder 1
    x = Conv2D(64, (3,3), padding='same', strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, (3,3), padding='same', strides=1)(x)
    shortcut = Conv2D(64, (1,1), padding='same', strides=1)(inputs) #shortcut using the identity matrix
    skip1 = x + shortcut #this referes to the skip connection for the decoder
    
    #Encoder 2 and 3
    skip2 = residual_block(skip1, 128, strides=2)
    skip3 = residual_block(skip2, 256, strides=2)
    
    #Bridge/Bottleneck
    b = residual_block(skip3, 512, strides=2)
    
    #Decoder 1, 2, 3
    x = decoder_block(b, skip3, 256)
    x = decoder_block(x, skip2, 128)
    x = decoder_block(x, skip1, 64)
    
    #Classifier
    outputs= Conv2D(1, (1,1), padding='same', activation='sigmoid')(x)
    
    #Model
    model = Model(inputs, outputs)
    
    #Metrics
    iou = BinaryIoU()
    
    #Compile
    model.compile(optimizer='adam', loss=loss_sum, metrics=['accuracy', iou])
    
    model.summary()
       
    return model