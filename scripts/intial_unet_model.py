def unet_model(img_height, img_width, channels):

    #Input
    inputs = Input((img_height, img_width, channels))
    inputs = Lambda(lambda x: x / 255)(inputs) #Normalize the pixels by dividing by 255

    #Encoder where we are extracting the features
    convolution1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    convolution1 = BatchNormalization()(convolution1) #other option is to do dropout by batch is faster 
    convolution1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(convolution1)
    pooling1 = MaxPooling2D((2, 2))(convolution1)
      
    convolution1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    convolution1 = BatchNormalization()(convolution1) 
    convolution1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(convolution1)
    pooling1 = MaxPooling2D((2, 2))(convolution1)
    
    convolution2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(pooling1)
    convolution2 = BatchNormalization()(convolution2)
    convolution2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(convolution2)
    pooling2 = MaxPooling2D((2, 2))(convolution2)
     
    convolution3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pooling2)
    convolution3 = BatchNormalization()(convolution3)
    convolution3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(convolution3)
    pooling3 = MaxPooling2D((2, 2))(convolution3)
     
    convolution4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pooling3)
    convolution4 = BatchNormalization()(convolution4)
    convolution4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(convolution4)
    pooling4 = MaxPooling2D(pool_size=(2, 2))(convolution4)

    #Bottleneck at the base of the U-net 
    convolution5 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(pooling4)
    convolution5 = BatchNormalization()(convolution5)
    convolution5 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(convolution5)
    
    #Decoder where we are indicating to the model the precise location of the features 
    transconv6 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')(convolution5)
    transconv6 = concatenate([transconv6, convolution4])
    convolution6 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(transconv6)
    convolution6 = BatchNormalization()(convolution6)
    convolution6 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(convolution6)
     
    transconv7 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(convolution6)
    transconv7 = concatenate([transconv7, convolution3])
    convolution7 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(transconv7)
    convolution7 = BatchNormalization()(convolution7)
    convolution7 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(convolution7)
     
    transconv8 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')(convolution7)
    transconv8 = concatenate([transconv8, convolution2])
    convolution8 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(transconv8)
    convolution8 = BatchNormalization()(convolution8)
    convolution8 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(convolution8)
     
    transconv9 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='same')(convolution8)
    transconv9 = concatenate([transconv9, convolution1], axis=3)
    convolution9 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(transconv9)
    convolution9 = BatchNormalization()(convolution9)
    convolution9 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(convolution9)
     
    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(convolution9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    jaccard = MeanIoU(2)
    
    #loss options include: binary_crossentropy, IoU Loss (Jaccard Index), dice loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[jaccard]) 
    
    model.summary()
    
    return model