Skip to content
Search or jump to…
Pull requests
Issues
Codespaces
Marketplace
Explore
 
@areisdorf45 
wilsontown
/
solar_panel_app
Public
Code
Issues
Pull requests
Actions
Projects
Security
Insights
solar_panel_app/Scripts/webtest.py /
@wilsontown
wilsontown Percentage calculation
Latest commit f03c09b yesterday
 History
 1 contributor
193 lines (140 sloc)  5.9 KB

import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, concatenate, Activation, Concatenate
from tensorflow.keras.metrics import IoU, BinaryIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from skimage import io
import googlemaps
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv

env_path = load_dotenv()
password = os.getenv("GOOGLE_MAPS")
gmaps = googlemaps.Client(key=password)

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters) #can be used as skip connection
    p = MaxPooling2D((2,2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters): #skip features are going to be the x returned from the encoder block
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
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

    model.compile(optimizer='adam', loss=loss_sum, metrics=['accuracy'])

    #model.summary()

    return model

@st.cache
def predict_mask (model, file, input_size):
    image = io.imread(file)
    image = image[:, :, :3]
    height = image.shape[0]
    width = image.shape[1]

    if height % input_size == 0:
        y_indices = np.arange(0, height, input_size)
    else:
        y_indices = np.arange(0, height - input_size, input_size)

    if width % input_size == 0:
        x_indices = np.arange(0, width, input_size)
    else:
        x_indices = np.arange(0, width - input_size, input_size)

    rows = len(y_indices)
    columns = len(x_indices)

    tiles = []
    for x in x_indices:
        for y in y_indices:
            split = image[y:(y + input_size), x:(x + input_size), :]
            array = np.array(split)
            tiles.append(array)

    predict_set = np.array(tiles)

    results = model.predict(predict_set)

    items = []
    for n in range (rows * columns):
        items.append(n)
    grid = np.array(items).reshape(columns, rows)

    tiled = []
    for n in range(columns):
        col = []
        for row in grid[n, :]:
            col.append(results[row])
        tiled.append(np.vstack(col))
    reconstruction = np.hstack(tiled)

    return reconstruction

model = build_unet(256, 256, 3)
model_weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_weights", "loss_sum_trainingset")
model.load_weights(model_weights_path)

tab1, tab2 = st.tabs(['From file', 'From googlemaps'])

try:

    st.sidebar.image('Images/logo.png')
    st.sidebar.markdown("""<span style="word-wrap:break-word;">Building footprint identification from satellite images""", unsafe_allow_html=True)

    with tab1:
        st.image('Images/banner.png')
        st.title('Solar panel segmentation')

        testfile = st.file_uploader('Please select a satellite image', type=['png', 'jpg', 'tif'])

        col1, col2 = st.columns(2)

        if testfile is not None:
            col1.image(testfile)

            result = predict_mask(model, testfile, 256)

            percent = (np.sum(result) / (result.shape[0] * result.shape[1])) * 100
            percent = round(percent, 1)

            col2.image(result)

            col2.text(f'Percentage roofspace: {percent} %')

    with tab2:
        st.image('Images/banner.png')
        st.title('Solar panel segmentation')

        place = st.text_input("Please enter a location (e.g. an address or a city)")

        if place is not None:
            geocode_result = gmaps.geocode(place)

            lat = geocode_result[0]['geometry']['location']['lat']
            lon = geocode_result[0]['geometry']['location']['lng']

            sat = f'https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&format=jpg&zoom=17&size=640x640&scale=2&maptype=satellite&key={password}'

            col1, col2 = st.columns(2)

            col1.image(sat)

            mask = predict_mask(model, sat, 256)

            percent = (np.sum(mask) / (mask.shape[0] * mask.shape[1])) * 100
            percent = round(percent, 1)

            col2.image(mask)
            col2.text(f'Percentage roofspace: {percent} %')

except:
    pass
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
solar_panel_app/webtest.py at master · wilsontown/solar_panel_app