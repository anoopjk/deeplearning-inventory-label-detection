import os
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense


os.chdir('C:/Users/Anoop/Desktop/job_applications/IFM/')
# path to the model weights files.
#weights_path = '../keras/examples/vgg16_weights.h5'
weights_path = 'DeepLearning/weights/vgg16_weights.h5'
top_model_weights_path ='DeepLearning/weights/bottleneck_fc_model.h5' 
# dimensions of our images.
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

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (3, img_width, img_height))
print('Model loaded.')


# build a classifier model to put on top of the convolutional model

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)
model = Model(input = base_model.input, output = top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale= 1., 
                             featurewise_center=True,
                             shear_range= 0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)
train_datagen.mean = np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)

test_datagen = ImageDataGenerator(rescale=1., 
                                  featurewise_center=True)
test_datagen.mean = np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)
# prepare data augmentation configuration
#train_datagen = ImageDataGenerator(
#    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

#test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
nb_val_samples=nb_validation_samples)

model.save_weights(weights_path)

#####################################################################################
# model prediction
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input 
#model.load_weights(weights_path)
img_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/testing/designtile.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.ImageDataGenerator()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred = model.predict(x)
print pred

