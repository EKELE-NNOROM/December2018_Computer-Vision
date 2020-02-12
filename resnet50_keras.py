#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 19:00:00 2019

@author: ekele
"""

from __future__ import print_function
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Add, Input, ZeroPadding2D, AveragePooling2D, GlobalMaxPooling2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import ELU
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import numpy as np

num_classes = 6
img_rows, img_cols = 64, 64

train_data_dir = 'fer2013/train'
validation_data_dir = 'fer2013/validation'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        shear_range=0.25,
        zoom_range=0.25,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(
        rescale=1./255)

batch_size = 16

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'rgb',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'rgb',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

# Identity block
def identity_block(X, f, filters):
#    conv_name_base = 'res' + str(stage) + block + '_branch'
#    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    # Shortcut branch
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size= (1, 1), padding = 'valid', kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size= (f, f), padding = 'same', kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    
    # Third component of main path
    X = Conv2D(filters = F3, kernel_size= (1, 1), padding = 'valid', kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    
    # Adding the shortcut branch to main path
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)
    
    return X

tf.reset_default_graph()

# testing identity block
with tf.Session() as sess:
    A_features = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_features, f = 2, filters = [2, 4, 6])
    sess.run(tf.global_variables_initializer())
    # 0 for testing, 1 for training
    y_hat = sess.run([A], feed_dict = {A_features: X, K.learning_phase(): 0}) 
    print("y_hat = " + str(y_hat[0][1][1][0]))
    

def convolutional_block(X, f, filters, s = 2)   :
    
    F1, F2, F3 = filters
    
    # Shortcut branch
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size= (1, 1), padding = 'valid', strides = (s, s), kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size= (f, f), padding = 'same', strides = (1, 1), kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    
    # Third component of main path
    X = Conv2D(filters = F3, kernel_size= (1, 1), padding = 'valid', strides = (1, 1), kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    
    # Adding a conv layer and batch norm layer to Shortcut so as to make for equal dimension with X
    X_shortcut = Conv2D(filters = F3, kernel_size= (1, 1), padding = 'valid', strides = (s, s), kernel_initializer = 'glorot_uniform')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
    X_shortcut = Activation('elu')(X_shortcut)
    
    # Adding the shortcut branch to main path
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)
    
    return X
   
tf.reset_default_graph()    

# testing convolutional block
with tf.Session() as sess:
    A_features = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_features, f = 2, filters = [2, 4, 6])
    sess.run(tf.global_variables_initializer())
    # 0 for testing, 1 for training
    y_hat = sess.run([A], feed_dict = {A_features: X, K.learning_phase(): 0}) 
    print("y_hat = " + str(y_hat[0][1][1][0]))
    
    
def ResNet50(input_shape = (img_rows, img_cols, 3), classes = num_classes):
    X_input = Input(input_shape)
    
    # Zero-padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = 'glorot_uniform')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('elu')(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)
    
    # Stage 2 
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, f = 3, filters = [64, 64, 256])
    X = identity_block(X, f = 3, filters = [64, 64, 256])
    
    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_block(X, f = 3, filters = [128, 128, 512])
    X = identity_block(X, f = 3, filters = [128, 128, 512])
    X = identity_block(X, f = 3, filters = [128, 128, 512])
    
    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    X = identity_block(X, f = 3, filters = [256, 256, 1024])
    
    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 3)
    X = identity_block(X, f = 3, filters = [512, 512, 2048])
    X = identity_block(X, f = 3, filters = [512, 512, 2048])
    
    # Average Pooling
    X = AveragePooling2D((2, 2))(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', kernel_initializer = 'glorot_uniform')(X)
    
    model = Model(inputs = X_input, outputs = X)
    return model
    
    
model = ResNet50(input_shape = (img_rows, img_cols, 3), classes = num_classes)
print(model.summary())

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("TrainedModel/resnet50.h5",
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
              optimizer='adam',
              metrics=['accuracy'])

nb_train_samples = 28273
nb_validation_samples = 3534
epochs = 30

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
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
