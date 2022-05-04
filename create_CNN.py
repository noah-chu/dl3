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



data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')
image_count = len(list(data_dir.glob('*/*.png')))
I = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)

s = plt.imread(str(I[1]))

plt.imshow(s)
plt.show()

# create training set
x_train =[]
y_train=[]
for i in range(20):
  if i < 10:
    s = plt.imread(str(I[i])) 
    x_train.append(s)
  else:
    s = plt.imread(str(I[i])) 
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


data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_test')
image_count = len(list(data_dir.glob('*/*.png')))
I = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
s = cv2.imread(str(I[0]))
s.shape

plt.imshow(s)
plt.show()

s.shape
s


x_test =[]
y_test=[]
for i in range(20):
  if i < 10:
    s = plt.imread(str(I[i])) 
    x_test.append(s)
  else:
    s = plt.imread(str(I[i])) 
    y_test.append(s)


x_test = np.asarray(x_test)
x_test = x_train.astype('float32')
y_test = np.asarray(y_test)
y_test = y_test.astype('float32')
#x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = y_test / 255.0
#x_test = x_test / 255.0

x_train[0].shape



x_train.shape

x_train = x_train.reshape((x_train.shape[0], 516, 1083, 3))
x_test = x_test.reshape((x_test.shape[0], 516, 1083, 3))
plt.imshow(x_train[3])
plt.show()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


############### CNN ################

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
model = create_cnn(1083,516,3)



model.summary()
model.compile(loss='mean_absolute_error',optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
history = model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test), batch_size=32, epochs=1)



# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model



# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories