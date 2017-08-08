# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 01:05:43 2017

@author: Anoop
"""
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense



def vgg16_finetuned():
    # dimensions of our images.
    img_width, img_height = 224, 224
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (3, img_width, img_height))
    print('Model loaded.')
    
    
    # build a classifier model to put on top of the convolutional model
    
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    # add the model on top of the convolutional base
    model = Model(input = base_model.input, output = top_model(base_model.output))
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
                  
    return model