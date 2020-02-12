#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 18:22:19 2019

@author: ekele
"""

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import ELU
import os

num_classes = 20
img_rows, img_cols = 32, 32

train_data_dir = 'simpsons/train'
validation_data_dir = 'simpsons/validation'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(
        rescale=1./255)

batch_size = 16

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

model = Sequential()

# !st Conv Block
model.add(Conv2D(filters=64, input_shape=(img_rows, img_cols,3), kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 2nd Conv Block
model.add(Conv2D(filters=64, input_shape=(img_rows, img_cols,3), kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Max Pool Block
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 3rd Conv Block
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 4th Conv Block
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Max Pool Block
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 5th Conv Block
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# 6th Conv Block
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Max Pool Block
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 1st FC - Fully connected layer
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# 2nd FC Layer
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# top layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("simpsons_LITTLE_VGG.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

nb_train_samples = 19548
nb_validation_samples = 990
epochs = 20

history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples  // batch_size)

score = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size+1)
print('Test loss:', score[0])
print('Test accuracy', score[1])























