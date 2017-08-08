# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:51:04 2017

@author: Anoop
"""

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        pos/
            dog001.jpg
            dog002.jpg
            ...
        neg/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        pos/
            dog001.jpg
            dog002.jpg
            ...
        neg/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras import backend as K
# dimensions of our images.

os.chdir('C:/Users/Anoop/Desktop/job_applications/IFM/')



top_model_weights_path ='DeepLearning/weights/bottleneck_fc_model.h5' 
img_width, img_height = 224, 224

train_data_dir = 'label_dataset/train/'
validation_data_dir = 'label_dataset/validation/'
nb_pos_train = 227 
nb_neg_train = 1973
nb_train_samples = nb_pos_train + nb_neg_train
nb_pos_validation = 9
nb_neg_validation = 11
nb_validation_samples = nb_pos_validation + nb_neg_validation
validation_batch_size = 20

epochs = 5
batch_size = 20

#def save_bottlebeck_features():
                             
datagen = ImageDataGenerator(rescale= 1., featurewise_center = True)
datagen.mean =np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)


#datagen = ImageDataGenerator(rescale=1. / 255)
# build the VGG16 network
model = VGG16(include_top=False, weights='imagenet' )


generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_train = model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('DeepLearning/features/bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)
print('features for train samples saved')

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_validation = model.predict_generator(
    generator, nb_validation_samples // validation_batch_size)
np.save(open('DeepLearning/features/bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)
print('features for validation samples saved')        


#def train_top_model():

train_data = np.load(open('DeepLearning/features/bottleneck_features_train.npy', 'rb'))
#train_data = bottleneck_features_train 
train_labels = np.array(
    [0] * (nb_neg_train) + [1] * (nb_pos_train))

validation_data = np.load(open('DeepLearning/features/bottleneck_features_validation.npy', 'rb'))
#validation_data = bottleneck_features_validation
validation_labels = np.array(
    [0] * (nb_neg_validation) + [1] * (nb_pos_validation))

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)
