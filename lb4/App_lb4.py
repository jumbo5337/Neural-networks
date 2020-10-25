import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

def label_to_int(label):
    if label == 'DH':
        return 0
    elif label == 'SL':
        return 1
    elif label == 'NO':
        return 2


samples = np.zeros((310, 7))
i = 0
with open('column_3C.dat', 'r') as file:
    for line in file.readlines():
        split = line.split()
        label = label_to_int(split[6])
        str_arr = np.array(split[0:6])
        str_arr.astype(float)
        arr = np.append(str_arr, label)
        samples[i] =  arr
        i=i+1

np.random.shuffle(samples)
Y_labels = to_categorical(samples[:, -1])
X_samples = samples[:, :6]

activation = 'relu'
neurons = 50
model = models.Sequential()
model.add(layers.Dense(64, activation=activation, input_dim=6))
model.add(layers.Dense(64, activation=activation))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation=activation))
model.add(layers.Dense(8, activation=activation))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(epochs = 700, x=X_samples, y=Y_labels)


from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_samples)
print(confusion_matrix(Y_labels, y_pred))