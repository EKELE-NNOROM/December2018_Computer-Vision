#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:01:35 2019

@author: ekele
"""

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg16
import keras.backend as K

ntName = "es"

content_img = './images/willy_wonka_old.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    
def deprocess_image(x):
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    
    # BGR -> RGB
    x = x[:,:,::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
         

x = np.array(([1,2],[3,4])) 
x.shape
y = np.expand_dims(x, axis = 0) 
y.shape
y_s = np.expand_dims(x, axis = 2) 
y_s.shape

L = range(10)
list(L)
list(L[::2])

def content_loss(content, mixed):
    return K.sum(K.square(content - mixed))

def gram_matrix(image):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, mixed):
    S = gram_matrix(style)
    C = gram_matrix(mixed)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(
            )
