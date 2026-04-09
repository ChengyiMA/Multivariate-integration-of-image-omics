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


# In[30]:


batchSize = 880  
startEpoch = 0
numEpochs = 3200 
newSize = 512  
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
    return allIms


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

#Flatten
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


#cae.summary()


# In[9]:


# Load trained weights
if os.path.isfile('/data/pretrained_3999.weights.h5'):
    cae.load_weights('/data/pretrained_3999.weights.h5')


# In[32]:

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-5)

# Compile model
cae.compile(loss='mse', optimizer=optimizer)

# Model training
for i in range(startEpoch, startEpoch + numEpochs):
    imBatch = getBatch(batchSize, '/data/TCGA_BRCA_datasets')/255
    cae.fit(imBatch, imBatch, epochs=1, verbose=True, batch_size=5)

    # Save weights every 100 epoc
    if (i + 1) % 100 == 0:
        cae.save_weights(f'/data/pretrained_{i}.weights.h5')

