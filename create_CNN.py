from importlib.resources import path
from matplotlib.font_manager import json_load
import json
import os
import tensorflow as tf
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image
from keras.utils import np_utils
import keras
from yaml import parse
import glob

try:
  os.chdir('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')
  data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')#desktop
except FileNotFoundError:
  try:
    os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')
    data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')#laptop
  except FileNotFoundError:
    data_dir = pathlib.Path('G:\\My Drive\\MSBA\\Spring\\Deep Learning\\GA3\\SVHN_Padded_train')
data_dir

image_count = len(list(data_dir.glob('*/*.png')))
I = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
s = cv2.imread(str(I[1]),cv2.IMREAD_GRAYSCALE)
s.shape


plt.imshow(s)
plt.show()



# create training set
x_train =[]
y_train=[]
for i in range(20):
  if i < 10:
    s = cv2.imread(str(I[i]),cv2.IMREAD_GRAYSCALE) 
    x_train.append(s)
  else:
    s = cv2.imread(str(I[i]),cv2.IMREAD_GRAYSCALE)
    y_train.append(s)




  ##normalize data

x_train = np.asarray(x_train)
x_train = x_train.astype('float32')
y_train = np.asarray(y_train)
y_train = y_train.astype('float32')
#x_test = x_test.astype('float32')
x_train = x_train / 255.0
y_train = y_train / 255.0
#x_test = x_test / 255.0


##split data and then OHE
#y_train = x_train
#y_train = np_utils.to_categorical(y_train)

len(x_train)

try:
  os.chdir('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning')
  data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_test')#desktop
except FileNotFoundError:
  try:
    os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning')
    data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_test')#laptop
  except FileNotFoundError:
    data_dir = pathlib.Path('G:\\My Drive\\MSBA\\Spring\\Deep Learning\\GA3\\SVHN_Padded_test')
data_dir

image_count = len(list(data_dir.glob('*/*.png')))
I = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
s = cv2.imread(str(I[0]))
s.shape



x_test =[]
y_test=[]
for i in range(20):
  if i < 10:
    s = cv2.imread(str(I[i]),cv2.IMREAD_GRAYSCALE) 
    x_test.append(s)
  else:
    s = cv2.imread(str(I[i]),cv2.IMREAD_GRAYSCALE) 
    y_test.append(s)


x_test = np.asarray(x_test)
x_test = x_train.astype('float32')
y_test = np.asarray(y_test)
y_test = y_test.astype('float32')
#x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = y_test / 255.0
#x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)




############### CNN TRY 1 #################

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    inputs = tf.keras.Input(shape=inputShape)

	# loop over the number of filters
    for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
            if i == 0:
                x = inputs
            # CONV => RELU => BN => POOL
    
            x = tf.keras.layers.Conv2D(f, (3, 3), padding="same")(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(16)(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(6,activation = 'softmax')(x)
            
         
            if regress:
                x = tf.keras.Dense(1, activation="linear")(x)
	# construct the CNN
            model =tf.keras.Model(inputs, x)
	# return the CNN
            return model
        
tf.keras.backend.clear_session()
model = create_cnn(1083,516,1)



model.summary()
model.compile(loss='mean_absolute_percentage_error',optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
history = model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test), batch_size=32, epochs=1)



















################## CNN TRY 2 #############################

tf.keras.backend.clear_session()


inputs = tf.keras.layers.Input(shape=(516,1083,1), name='input') 
x = tf.keras.layers.Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same')(inputs)
#x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
#x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.BatchNormalization()(x)
   
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.BatchNormalization()(x)


x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
#x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.BatchNormalization()(x)
    
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
#x = tf.keras.layers.Dropout(0.2)(x)


x = tf.keras.layers.Dense(32, activation='relu')(x)
#x = tf.keras.layers.Dropout(0.3)(x)
#x = tf.keras.layers.BatchNormalization()(x)
yhat = tf.keras.layers.Dense(6, activation='softmax')(x)




model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()
#Compile model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
model.fit(x=x_train,y=y_train, batch_size=10, epochs=1) 











################# CNN TRY 3 ####################



inputs = tf.keras.layers.Input(shape=(516,1083,1), name='input') 
#Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(inputs)
#MaxPooling2D: pool_size is window size over which to take the max
x = tf.keras.layers.MaxPooling2D(pool_size = 6, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 6, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 6, strides = 2, padding = "valid")(x)
#dense layers expect 1D array of features for each instance so we need to flatten.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
yhat = tf.keras.layers.Dense(6, activation = 'softmax')(x)





model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()
#Compile model
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
model.fit(x=x_train,y=y_train, batch_size=10, epochs=1) 








########### Try 4 ######################



inputs = tf.keras.layers.Input(shape=(516,1083,1), name='input') 
#Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(inputs)
#MaxPooling2D: pool_size is window size over which to take the max
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
#dense layers expect 1D array of features for each instance so we need to flatten.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
yhat = tf.keras.layers.Dense(6, activation = 'softmax')(x)

#Why do we stack two convolutional layers followed by a pooling layer, as opposed to having each convolutional layer followed by a pooling layer?
# Answer: every convolutional layer creates a number of feature maps (e.g,64) that are individually connected to the previous layer.
# By stacking two convolutional layers before inserting a pooling layer we allow the second convolutional layer to learn from the noisy signal, as opposed to the clean signal.

model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()
#Compile model
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
model.fit(x=x_train,y=y_train, batch_size=32, epochs=1) 
