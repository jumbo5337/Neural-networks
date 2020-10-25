import random

import matplotlib.pyplot as plt
import numpy as np
import pyreadr
from matplotlib import rcParams
from tensorflow import random as tf_rnd
from tensorflow.keras import layers
from tensorflow.keras import models

## Параметры размеров окна matplotlib
rcParams['figure.figsize'] = (10.0, 5.0)

np.random.seed(77)
tf_rnd.set_seed(77)
random.seed(77)

rda = pyreadr.read_r('debitcards.rda') # also works for Rds, rda
# Monthly retail debit card usage in Iceland (million ISK). January 2000 - August 2013.
df = rda['debitcards']
raw_data = np.reshape(np.asarray(df[:-8]), (156))

len = len(raw_data)
base_data = np.zeros((len-12, 13))
for i in range(0,(len-12)):
   base_data[i, ] = raw_data[i:(i+13)]

activation = 'relu'
neurons = 50
model = models.Sequential()
model.add(layers.Dense(neurons, activation=activation, input_dim=12))
model.add(layers.Dense(neurons, activation=activation))
model.add(layers.Dense(neurons, activation=activation))
model.add(layers.Dense(neurons * 2, activation=activation))
model.add(layers.Dense(neurons * 2, activation=activation))
model.add(layers.Dense(neurons * 2, activation=activation))
model.add(layers.Dense(neurons * 2, activation=activation))
model.add(layers.Dense(neurons * 2, activation=activation))
model.add(layers.Dense(neurons, activation=activation))
model.add(layers.Dense(20, activation=activation))
model.add(layers.Dense(5, activation=activation))
model.add(layers.Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(epochs = 700, x=base_data[:,0:12], y=base_data[:,-1])


pred_arr = np.reshape(model.predict(base_data[:,0:12]), 144)

predictions = np.zeros(156)
predictions[12:] = pred_arr

print(predictions)


plt.plot(raw_data, '-', color='green', label='Real Data', markersize=3)
plt.plot(predictions, '--', color='red', label='Predictions', markersize=3)
plt.legend()
plt.show()


