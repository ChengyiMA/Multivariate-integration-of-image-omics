#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, random_rotation, array_to_img
import tensorflow.keras.preprocessing.image as image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

imagePath = ''
pretrained = '/data/pretrained_3999.weights.h5'
outputName = '/data/CAE_training_output' # create this folder before running



# In[30]:


batchSize = 880  
numEpochs = 3000 
newSize  = 512 
sampSize = 128 
filterSize = 5 
poolSize = 2 

# In[28]:


def getBatch(batchSize, path):
    allPaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    allIms = np.zeros((batchSize, sampSize, sampSize, 3))
    for i in range(batchSize):
        #f1 = int(np.sign(np.random.rand() - 0.5))
        #f2 = int(np.sign(np.random.rand() - 0.5))
        im = load_img(allPaths[i]).resize((newSize, newSize))
        r = int(np.random.rand() * (newSize - sampSize))
        c = int(np.random.rand() * (newSize - sampSize))
        im = im.crop((r, c, r + sampSize, c + sampSize))
        im_array = img_to_array(im)
        imageNew = expand_dims(im_array, 0)
        imageDataGen = ImageDataGenerator(rotation_range=10)
        iterator = imageDataGen.flow(imageNew, batch_size=1)
        batch = iterator.next()
        rotated_image = batch[0]
        #im_array = image.random_rotation(im_array, 5)
        #im_array = img_to_array(im) / 255.0
        #im = random_rotation(im, 5)
        
        allIms[i, :, :, :] = rotated_image
    return allIms/255
    
def getOne(batchSize, imPath):
	allIms = np.zeros((batchSize, sampSize, sampSize, 3))
	for i in range(batchSize):
		#f1 = int(np.sign(np.random.rand() - .5))
		#f2 = int(np.sign(np.random.rand() - .5))
		im = load_img(imPath).resize((newSize, newSize))
		r  = int(np.random.rand() * (newSize - sampSize))
		c  = int(np.random.rand() * (newSize - sampSize))
		im = im.crop((r, c, r + sampSize, c + sampSize))
		im_array = img_to_array(im)
		imageNew = expand_dims(im_array, 0)
		imageDataGen = ImageDataGenerator(rotation_range=10)
		iterator = imageDataGen.flow(imageNew, batch_size=1)
		batch = iterator.next()
		rotated_image = batch[0]
		#allIms[i, :, :, :] = image.img_to_array(im)
		allIms[i, :, :, :] = rotated_image
	return allIms/255    


# In[5]:


# initialize cae
cae = Sequential()


# In[6]:


# Encoder
cae.add(Conv2D(8, (filterSize, filterSize), input_shape=(sampSize, sampSize, 3), padding='same'))
cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
cae.add(Activation('relu'))

cae.add(Conv2D(16, (filterSize, filterSize), padding='same'))
cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
cae.add(Activation('relu'))

cae.add(Conv2D(32, (filterSize, filterSize), padding='same'))
cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
cae.add(Activation('relu'))

cae.add(Conv2D(64, (filterSize, filterSize), padding='same'))
cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
cae.add(Activation('relu'))

cae.add(Conv2D(128, (filterSize, filterSize), padding='same'))
cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
cae.add(Activation('relu'))


# In[7]:


# Flatten
cae.add(Flatten())
cae.add(Dense(1024, activation='relu'))
cae.add(Dense(128 * 4 * 4, activation='relu'))
cae.add(Reshape((4, 4, 128)))
cae.add(Activation('relu'))


# In[8]:


# Decoder
cae.add(UpSampling2D(size=(poolSize, poolSize)))
cae.add(Conv2D(64, (filterSize, filterSize), padding='same'))
cae.add(Activation('relu'))

cae.add(UpSampling2D(size=(poolSize, poolSize)))
cae.add(Conv2D(32, (filterSize, filterSize), padding='same'))
cae.add(Activation('relu'))

cae.add(UpSampling2D(size=(poolSize, poolSize)))
cae.add(Conv2D(16, (filterSize, filterSize), padding='same'))
cae.add(Activation('relu'))

cae.add(UpSampling2D(size=(poolSize, poolSize)))
cae.add(Conv2D(8, (filterSize, filterSize), padding='same'))
cae.add(Activation('relu'))

cae.add(UpSampling2D(size=(poolSize, poolSize)))
cae.add(Conv2D(3, (filterSize, filterSize), padding='same'))
cae.add(Activation('sigmoid'))


# In[33]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-5)

# Compile model
cae.compile(loss='mse', optimizer='adam')

# Initialize encoder
encode = Sequential()

encode.add(Conv2D(
    filters=8,
    kernel_size=(filterSize, filterSize),
    input_shape=(sampSize, sampSize, 3),  
    padding='same'))  
encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
encode.add(Activation('relu'))

encode.add(Conv2D(
    filters=16,
    kernel_size=(filterSize, filterSize),
    padding='same'))
encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
encode.add(Activation('relu'))

encode.add(Conv2D(
    filters=32,
    kernel_size=(filterSize, filterSize),
    padding='same'))
encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
encode.add(Activation('relu'))

encode.add(Conv2D(
    filters=64,
    kernel_size=(filterSize, filterSize),
    padding='same'))
encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
encode.add(Activation('relu'))

encode.add(Conv2D(
    filters=128,
    kernel_size=(filterSize, filterSize),
    padding='same'))
encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
encode.add(Activation('relu'))

encode.add(Flatten())
encode.add(Dense(1024))
encode.add(Activation('relu'))

encode.layers[0].set_weights(cae.layers[0].get_weights())  # For the first Conv2D layer
encode.layers[3].set_weights(cae.layers[3].get_weights())  # For the second Conv2D layer
encode.layers[6].set_weights(cae.layers[6].get_weights())  # For the third Conv2D layer
encode.layers[9].set_weights(cae.layers[9].get_weights())  # For the fourth Conv2D layer
encode.layers[12].set_weights(cae.layers[12].get_weights())  # For the fifth Conv2D layer
encode.layers[16].set_weights(cae.layers[16].get_weights())  # For the Dense layer

# Compile the encoder
encode.compile(loss='mse', optimizer='adam')

# Initialize decoder
decode = Sequential()

# Dense Layer (First layer of decoder)
decode.add(Dense(units=128 * 4 * 4, input_shape=(1024,), activation='relu'))
decode.layers[-1].set_weights(cae.layers[17].get_weights())

# Reshape Layer
decode.add(Reshape((4, 4, 128)))  # Reshape to (batch_size, 4, 4, 128)
decode.add(Activation('relu'))

# UpSampling + Conv2D Layers
decode.add(UpSampling2D(size=(poolSize, poolSize)))
decode.add(Conv2D(filters=64, kernel_size=(filterSize, filterSize), padding='same', activation=None))
decode.layers[-1].set_weights(cae.layers[21].get_weights())  # Set weights for Conv2D layer
decode.add(Activation('relu'))

decode.add(UpSampling2D(size=(poolSize, poolSize)))
decode.add(Conv2D(filters=32, kernel_size=(filterSize, filterSize), padding='same', activation=None))
decode.layers[-1].set_weights(cae.layers[24].get_weights())  # Set weights for Conv2D layer
decode.add(Activation('relu'))

decode.add(UpSampling2D(size=(poolSize, poolSize)))
decode.add(Conv2D(filters=16, kernel_size=(filterSize, filterSize), padding='same', activation=None))
decode.layers[-1].set_weights(cae.layers[27].get_weights())  # Set weights for Conv2D layer
decode.add(Activation('relu'))

decode.add(UpSampling2D(size=(poolSize, poolSize)))
decode.add(Conv2D(filters=8, kernel_size=(filterSize, filterSize), padding='same', activation=None))
decode.layers[-1].set_weights(cae.layers[30].get_weights())  # Set weights for Conv2D layer
decode.add(Activation('relu'))

decode.add(UpSampling2D(size=(poolSize, poolSize)))
decode.add(Conv2D(filters=3, kernel_size=(filterSize, filterSize), padding='same', activation=None))
decode.layers[-1].set_weights(cae.layers[33].get_weights())  # Set weights for Conv2D layer
decode.add(Activation('sigmoid'))

decode.compile(loss='mse', optimizer='adam')

allPaths = [os.path.join(imagePath, f) for f in os.listdir(imagePath) if f.endswith(('.jpg', '.png', '.jpeg'))]

repAll = np.zeros((len(allPaths), 1024))

for i in range(len(repAll)):
	print(i)
	for co in range(10):
		repAll[i, :] += sum(encode.predict(getOne(100, allPaths[i])))
	repAll[i, :] /= 1000.
np.save('/extracted_image_feature/repAll.npy', repAll)




