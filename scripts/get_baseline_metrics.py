import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from tensorflow.keras.metrics import MeanIoU

def get_baseline_metrics(directory):

    accuracies = []
    IoUs = []

    files = os.listdir(directory)

    baseline_prediction = np.zeros((65536))

    for file in files:
        image = io.imread(f'{directory}/{file}')
        image_flat = image.reshape(65536)

        metric = MeanIoU(2)
        metric.update_state(image_flat / 255, baseline_prediction)
        IoUs.append(metric.result().numpy())

        accuracy = accuracy_score(image_flat / 255, baseline_prediction)
        accuracies.append(accuracy)

    return np.array(accuracies).mean(), np.array(IoUs).mean()
