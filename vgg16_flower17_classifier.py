#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:18:27 2019

@author: ekele
"""

from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

# set dimensions of image and number of classes at output
img_rows, img_cols = 224, 224
num_classes = 17

# import VGG16 the top layer
vgg16 = VGG16(weights = 'imagenet',
                       include_top = False,
                       input_shape = (img_rows, img_cols, 3))

# freeze lower layers
for layer in vgg16.layers:
    layer.trainable = False
  
top_model = vgg16.output
top_model = Flatten(name = 'flatten')(top_model)
top_model = Dense(256, activation = 'relu')(top_model)
top_model = Dropout(0.3)(top_model)
FC_Head = Dense(num_classes, activation = 'softmax')(top_model)

model = Model(inputs = vgg16.input, outputs = FC_Head)

print(model.summary())

train_data_dir = '17_flowers/train'
validation_data_dir = '17_flowers/validation'
import os
os.listdir(validation_data_dir)
os.getcwd()

# data augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(
        rescale=1./255)

# batch size
train_batchsize = 16
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=train_batchsize,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=val_batchsize,
        class_mode='categorical')

trained_model_directory = "TrainedModel/"

if not os.path.exists(trained_model_directory):
    os.makedirs(trained_model_directory)
    
filename = "flowers_vgg_17-{epoch:02d}-{val_acc:.2f}.h5"
#os.path.join(trained_model_directory, filename)

# Setting callbacks
checkpoint = ModelCheckpoint(os.path.join(trained_model_directory, filename),
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1)

callbacks = [earlystop, checkpoint]

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 1190
nb_validation_samples = 170

epochs = 10
batch_size = 16

# train model
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

model.save("flowers_vgg_17.h5")

