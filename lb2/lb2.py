import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import random as tf_rnd
import random

np.random.seed(77)
tf_rnd.set_seed(77)
random.seed(77)

## Параметры размеров окна matplotlib
rcParams['figure.figsize'] = (10.0, 5.0)

x_arr = np.arange(-10, 10, 0.5)
y_arr = np.tile([1, 1, 1, 1, 0, 0, 0, 0], 5)

plt.plot(x_arr, y_arr, '-', color='red', label='Real values', markersize=5)
plt.show()
plt.close()

threshold = 0.0001
keras_lf = losses.MeanSquaredError()


class signal_nn:
    def __init__(self, neurons=50, activation='relu', threshold=0.000001):
        self.threshold = threshold
        self.model = models.Sequential()
        self.model.add(layers.Dense(neurons, activation=activation, input_dim=1))
        self.model.add(layers.Dense(neurons, activation=activation))
        self.model.add(layers.Dense(neurons, activation=activation))
        self.model.add(layers.Dense(neurons * 2, activation=activation))
        self.model.add(layers.Dense(neurons * 2, activation=activation))
        self.model.add(layers.Dense(neurons * 2, activation=activation))
        self.model.add(layers.Dense(neurons, activation=activation))
        self.model.add(layers.Dense(20, activation=activation))
        self.model.add(layers.Dense(5, activation=activation))
        self.model.add(layers.Dense(1))
        self.model.compile(loss=self.custom_mse, optimizer='rmsprop')

    def custom_mse(self, y_true, y_pred):
        res = keras_lf(y_true, y_pred)
        return tf.where(res < self.threshold, 0.0, res)

    def fit(self, epochs, x, y):
        return self.model.fit(x, y, epochs=epochs, batch_size=1)

    def predict(self, x):
        return self.model.predict(x)

snn_tr_0001 = signal_nn(threshold= 0.0001)
snn_tr_001 = signal_nn(threshold= 0.001)
snn_tr_01 = signal_nn(threshold= 0.01)

snn_tr_01.fit(epochs=700, x = x_arr, y=y_arr)
snn_tr_001.fit(epochs=700, x = x_arr, y=y_arr)
snn_tr_0001.fit(epochs=700, x = x_arr, y=y_arr)

tr_01_arr = snn_tr_01.predict(x_arr)
tr_001_arr = snn_tr_001.predict(x_arr)
tr_0001_arr = snn_tr_0001.predict(x_arr)

plt.plot(x_arr, y_arr, '-', color='black', label='Real values', markersize=6)
plt.plot(x_arr, tr_01_arr, linestyle = ':', color='blue', label='Predict : tr = 0.01', markersize=3)
plt.plot(x_arr, tr_001_arr, '-.', color='green', label='Predict : tr = 0.001', markersize=3)
plt.plot(x_arr, tr_001_arr, '--', color='red', label='Predict : tr = 0.0001', markersize=3)
plt.legend()
plt.show()
