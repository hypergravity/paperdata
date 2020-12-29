import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nsample = 10000
ndim = 2
nhidden1 = nhidden2 = 1000

x = np.random.normal(0, 1, (nsample, ndim))
y = x[:, 0] ** 2 + x[:, 1] ** 2 + np.sin(x[:, 1])

x_train, x_test, pb_train, pb_test = train_test_split(x, y)

model = Sequential()
model.add(Dense(nhidden1, input_shape=(x_train.shape[1],), activation="sigmoid"))
model.add(BatchNormalization())
model.add(Dense(nhidden2, activation="sigmoid"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))

opt = SGD(0.000001)
model.compile(loss='mse',
              optimizer=opt,
              metrics=['mse'])

model.summary()

history = model.fit(x_train, pb_train,
                    batch_size=100,
                    epochs=1000,
                    validation_split=0.1,
                    initial_epoch=0,
                    validation_data=(x_test, pb_test))
