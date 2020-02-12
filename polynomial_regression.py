#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:36:07 2019

@author: ekele
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)
points = 500
X = np.linspace(-3, 3, num=points)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

plt.scatter(X, y)

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=1))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.01))
model.fit(X, y, epochs=50)

predictions = model.predict(X)
plt.scatter(X, y)
plt.plot(X, predictions, 'ro')
plt.show()