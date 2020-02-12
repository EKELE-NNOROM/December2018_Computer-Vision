#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:36:00 2019

@author: ekele
"""


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2

np.random.seed(0)

with open('german-traffic-signs/train.p', 'rb') as fhand:
    train_data = pickle.load(fhand)
    
with open('german-traffic-signs/valid.p', 'rb') as fhand:
    validation_data = pickle.load(fhand)
    
with open('german-traffic-signs/test.p', 'rb') as fhand:
    test_data = pickle.load(fhand)
    
print(type(train_data))
print(train_data.keys())

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = validation_data['features'], validation_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels"
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels"
assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32, 32, 3"
assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32, 32, 3"
assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32, 32, 3"

data = pd.read_csv('german-traffic-signs/signnames.csv')
print(data)

num_of_samples = []

data.describe()
data.info()

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10, 40))
fig.tight_layout()

for i in range(cols):
    for j, row in data.iterrows() :
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + '-' + row['SignName'])
            num_of_samples.append(len(x_selected))
            

#for i, row in data.iterrows():
#    x_selected = X_train[y_train == i]
#    for j in range(cols):
#        axs[i][j].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
#        axs[i][j].axis("off")
#        if j == 2:
#            axs[i][j].set_title(str(j) + '-' + row['SignName'])
#            num_of_samples.append(len(x_selected))
          
#print(y_train)
#print(num_of_samples)
#len(num_of_samples)

plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()
classes = range(0, num_classes)
list(classes)
df = pd.DataFrame({'Class number': classes,
                   'Number of images': num_of_samples})
    
len(num_of_samples)


plt.imshow(X_train[1200])
plt.axis("off")
print(X_train[1200].shape)
print(y_train[1200])

def gray_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

img = gray_scale(X_train[1000])
plt.imshow(img)
plt.axis("off")
print(img.shape)

# Using opencv
#cv2.imshow('GrayScale', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Histogram equalization - to standardize the lightening in all of the images
def equalize(img):
    # only accepts grayscale images
    img = cv2.equalizeHist(img)
    return img

img = equalize(img)
plt.imshow(img)
plt.axis("off")
print(img.shape)

def preprocessing(img):
    img = gray_scale(img)
    img = equalize(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

# plot a random training image
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis("off")
print(X_train.shape)

# adding depth to our image
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

y_train = to_categorical(y_train, num_classes=43)
y_val = to_categorical(y_val, num_classes=43)
y_test = to_categorical(y_test, num_classes=43)

def modified_lenet_model():
    model = Sequential()
    #before - model.add(Conv2D(30, (5, 5), input_shape=X_train.shape[1:]))
    # in order to improve accuracy
    model.add(Conv2D(60, (5, 5), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(60, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # before - model.add(Conv2D(15, (3, 3)))
    # in order to improve accuracy, increase number of filters
    model.add(Conv2D(30, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(30, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model
    
model = modified_lenet_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_val, y_val), batch_size = 400, verbose = 1, shuffle = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

score = model.evaluate(X_test, y_test, verbose=1)
# Score is at index 0
print("Test score", score[0])

# Accuracy is at index 1
print("Test Accuracy", score[1])

model.save("TrainedModel/traffic_sign_detection.h5")

import requests
from PIL import Image

url = "https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg"
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))
print(img.shape)

# Reshape
# add new dimension at beginning
img = np.expand_dims(img, axis=0)
# add new dimension at end
img = np.expand_dims(img, axis=-1)
print(img.shape)

#Test image
print("predicted sign: "+ str(model.predict_classes(img)))





















