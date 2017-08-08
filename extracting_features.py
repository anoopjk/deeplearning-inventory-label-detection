# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:06:11 2017

@author: Anoop
"""
"""
In summary, this is our directory structure:
```
data/
    train/
        pos/
            ...
        neg/
            ...
    validation/
        pos/
            ...
        neg/
            ...
```
"""

import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, \
img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.constraints import maxnorm
#dimensions of images
img_width, img_height = 150, 150


os.chdir('C:/Users/Anoop/Desktop/job_applications/IFM/')
train_data_dir = 'label_dataset/train'
validation_data_dir = 'label_dataset/validation'
nb_train_samples = 227+ 1974
nb_validation_samples = 13
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

#define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten()) #this converts our 3D feature maps to 1D feature
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
  

# training data augmentation configuration

train_datagen = ImageDataGenerator(
            width_shift_range= 0.2,
            height_shift_range =0.2,
            rescale = 1./255,
            shear_range =0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode='nearest')
            
test_datagen = ImageDataGenerator(rescale = 1./255)
            


# testing data augmentation configuration only rescaling

# this is a generator that will read pictures found in subfolders of 
#label_dataset/', and indefinitely generate batches of augmented data

train_generator = train_datagen.flow_from_directory(
                train_data_dir, #this is target directory
                target_size = (img_width, img_height), # resizing images to 150x150
                batch_size = batch_size,
                class_mode = 'binary' )#for binary labels #since we use binary_crossentropy,

#this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
                validation_data_dir, 
                target_size= (img_width, img_height),
                batch_size= 10,
                class_mode= 'binary')
                

            
model.fit_generator(
        train_generator,
        steps_per_epoch= nb_train_samples // batch_size,
        epochs= epochs,
        validation_data=validation_generator,
        validation_steps= nb_validation_samples // 1 )
        
model.save_weights('first_try.h5') #always save the weights after training or during training            
