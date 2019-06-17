'''
Created on Jan 24, 2017

@author: aelsalla
'''

#Simple two layer neural net minimizing the mean squared value. I am trying to switch to R2 loss later (see my attempt in the code)
import kagglegym

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, GlobalMaxPooling1D, Embedding, LSTM
from keras.utils import np_utils
from keras import backend as K

import numpy as np

# Create environment
env = kagglegym.make()

# Get first observation
observation = env.reset()

# Data
mean_vals = observation.train.mean()
traindf = observation.train.drop(axis=1, labels=["id", "timestamp"]).fillna(mean_vals)

Y_train = traindf["y"]
X_train = traindf.drop(axis=1, labels=["y"])


# Model
input_shape=108
#input_shape=X_train.shape
LEARNING_RATE = 0.001


# LSTM
input_dim=108
output_dim = 32
input_length = 10

# number of convolutional filters to use
nb_filters = 32

# convolution kernel size
filter_length = 1

hidden_dims = 100

batch_size = 32

nb_epoch = 2

'''
model = Sequential()

model.add(Convolution1D(nb_filters, kernel_size,
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution1D(nb_filters, kernel_size, kernel_size[1]))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
'''

print('Build model...')
model = Sequential()

model.add(LSTM(output_dim=output_dim, input_dim=input_dim, input_length=input_length))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
'''
model.add(Convolution1D(nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))

'''
# we add a Convolution1D, which will learn nb_filter
'''
model.add(Convolution1D(input_shape=input_shape,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))

# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
'''
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Activation('relu'))
model.add(Dense(1))

# Objective
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)



# Testing
#_________

# Integrate to kaggle
rewards = []
n = 0
while True:
    target = observation.target
    features = observation.features.drop(axis=1, labels=["id", "timestamp"])
    output = model.predict(features.values)
    target.loc[:, 'y'] = output
    # Fill in perfect actions
    #perfect_action = env.df[env.df["timestamp"] == observation.features["timestamp"][0]][["id", "y"]].reset_index(drop=True)
    #target.loc[:, 'y'] = perfect_action
    observation, reward, done, info = env.step(target)
    if done:
        break
    rewards.append(reward)
    n = n + 1
print(info)
print(n)
print(rewards[0:15])

