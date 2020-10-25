import matplotlib.pyplot as plt
import numpy as np
from numpy import arctan, sin
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import random as tf_rnd
import random

np.random.seed(77)
tf_rnd.set_seed(77)
random.seed(77)

fun = lambda x: 4 * arctan(x) + sin(x)
x_arr = np.asarray(np.random.sample(1000) * 10)
y_arr = np.asarray([fun(x) for x in x_arr])
plt.plot(x_arr, y_arr, '*', color='red', label='Real values', markersize = 5)
plt.show()
plt.close()

neurons = 20
epochs = 1500

tanh_20nn = models.Sequential()
tanh_20nn.add(layers.Dense(neurons, activation='tanh', input_dim=1))
tanh_20nn.add(layers.Dense(neurons/4, activation='tanh'))
tanh_20nn.add(layers.Dense(1))
tanh_20nn.compile(loss='mean_squared_error', optimizer='rmsprop')
tanh_20nn.fit(x_arr, y_arr, epochs=epochs, batch_size=128)

# tanh_40nn = models.Sequential()
# tanh_40nn.add(layers.Dense(neurons*2, activation='tanh', input_dim=1))
# tanh_40nn.add(layers.Dense(1))
# tanh_40nn.compile(loss='mean_squared_error', optimizer='rmsprop')
# tanh_40nn.fit(x_arr, y_arr, epochs=epochs, batch_size=128)
#
# tanh_50nn = models.Sequential()
# tanh_50nn.add(layers.Dense(neurons+30, activation='tanh', input_dim=1))
# tanh_50nn.add(layers.Dense(1))
# tanh_50nn.compile(loss='mean_squared_error', optimizer='rmsprop')
# tanh_50nn.fit(x_arr, y_arr, epochs=epochs, batch_size=128)


# relu_nn = models.Sequential()
# relu_nn.add(layers.Dense(neurons, activation='relu', input_dim=1))
# relu_nn.add(layers.Dense(1))
# relu_nn.compile(loss='mean_squared_error', optimizer='rmsprop')
# relu_nn.fit(x_arr, y_arr, epochs=epochs, batch_size=128)
#
# sgm_nn = models.Sequential()
# sgm_nn.add(layers.Dense(neurons, activation='sigmoid', input_dim=1))
# sgm_nn.add(layers.Dense(1))
# sgm_nn.compile(loss='mean_squared_error', optimizer='rmsprop')
# sgm_nn.fit(x_arr, y_arr, epochs=epochs, batch_size=128)


