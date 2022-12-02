import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf

def predict_mask (model, file, input_size):
    image = io.imread(file)

    height = image.shape[0]
    width = image.shape[1]

    y_indices = np.arange(0, height - input_size, input_size)
    rows = len(y_indices)
    x_indices = np.arange(0, width - input_size, input_size)
    columns = len(x_indices)

    tiles = []
    for x in x_indices:
        for y in y_indices:
            split = image[y:(y + input_size), x:(x + input_size), :]
            array = np.array(split / 255)
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
