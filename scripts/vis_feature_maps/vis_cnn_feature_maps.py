def vis_cnn_feature_maps(model, image):
    """input model and image, this function will display images after go through each CNN layer"""
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    image = tf.expand_dims(image, axis=0)
    feature_maps = feature_map_model.predict(image)
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if len(feature_map.shape) == 4: # Number of feature images/dimensions in a feature map of a layer 
            k = feature_map.shape[-1]  
            size=feature_map.shape[2]
            row = feature_map.shape[1]

            image_belt = np.array([[0]*k*size for i in range(row)])
            for i in range(k):
                feature_image = feature_map[0, :, :, i]  #first image of the batch for channel i
                feature_image -= feature_image.mean()
                feature_image /= feature_image.std()
                feature_image *=  64
                feature_image += 128
                feature_image = np.clip(feature_image, 0, 255)
                image_belt[:,i * size : (i + 1) * size] = feature_image

            scale = 20. / k
            plt.figure( figsize=(scale * k, scale) )
            plt.title ( layer_name )
            plt.grid  (False )
            plt.imshow(image_belt, aspect='auto')
