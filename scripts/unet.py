#Libaries 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import cv2 as cv

#Standard convolution block that we will pass into the encoder and decorder
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#Encoder that we will use to extract the different features from our images
def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters) #can be used as skip connection 
    p = MaxPooling2D((2,2))(x)
    return x, p

#Decoder that we will use to restore the spatial location of our images 
def decoder_block(inputs, skip_features, num_filters): #skip features are going to be the x returned from the encoder block
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Loss Functions:
def dice_loss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

#Our loss that combines binary crossentropy and dice loss
def loss_sum(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    o = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)

#Function will compile the unet architecture and output the model summary
def build_unet(img_height, img_width, channels):
    
    inputs = Input((img_height, img_width, channels))
    inputs = Lambda(lambda x: x / 255)(inputs) #Normalize the pixels by dividing by 255

    #Encoder - downscaling (creating features/filter)
    skip1, pool1 = encoder_block(inputs, 16)
    skip2, pool2 = encoder_block(pool1, 32) 
    skip3, pool3 = encoder_block(pool2, 64)
    skip4, pool4 = encoder_block(pool3, 128) 
    
    #Bottleneck or bridge between encoder and decoder
    b1 = conv_block(pool4, 256)
    
    #Decoder - upscaling (reconstructing the image and giving it precise spatial location)
    decoder1 = decoder_block(b1, skip4, 128)
    decoder2 = decoder_block(decoder1, skip3, 64)
    decoder3 = decoder_block(decoder2, skip2, 32)
    decoder4 = decoder_block(decoder3, skip1, 16)
    
    #Output
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(decoder4)
    model = Model(inputs, outputs)
    
    iou = BinaryIoU()
    
    model.compile(optimizer='adam', loss=loss_sum, metrics=['accuracy', iou])
    
    model.summary()
    
    return model
