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
import pandas as pd






########### read in labels ############
try:
       os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN_Padded_Train/train')
except FileNotFoundError:
       os.chdir("G:\\My Drive\\MSBA\\Spring\\Deep Learning\\GA3\\SVHN_Padded_train\\train")
jdata = pd.read_json(r'digitStruct.json')
jdata.head()

################  LABELS ########################
#initialize empty dictionary 
labels = {}
for row in range(jdata.shape[0]): #grab each row of the jdata frame
  rowlist = np.array([])
  for box in jdata.iloc[row,0]: #inspect each number of the picture
    #append it to an array
    rowlist = np.append(rowlist, box['label'])
  #add the labels of this entry to the dictionary
  labels[row] = rowlist
labels[0]






labels = {}
labels2 = {}
row = 0
for row in range(jdata.shape[0]): #grab each row of the jdata frame
  rowlist = np.ones(shape = (7,10), dtype = np.int64)*0
  rowlist2 = np.ones(shape = (7,10), dtype = np.int64)*0
  boxNum = 0
  for box in jdata.iloc[row,0]: #inspect each number of the picture
    #append it to an array
    rowlist = np.append(rowlist, tf.one_hot(int(box['label']), 10))
    rowlist2[boxNum] = tf.one_hot(int(box['label']), 10)
    boxNum += 1
  #add the labels of this entry to the dictionary
  rowlist2[6] = tf.one_hot((boxNum), 10)
  labels[row] = rowlist
  labels2[row] = rowlist2
labels2[0]





num_train = 100
num_test = 50
########## read in y_train for each number ###########


y_train1=np.empty(shape = (num_train,10))
y_train2=np.empty(shape = (num_train,10))
y_train3=np.empty(shape = (num_train,10))
y_train4=np.empty(shape = (num_train,10))
y_train5=np.empty(shape = (num_train,10))
y_train6=np.empty(shape = (num_train,10))
y_train7=np.empty(shape = (num_train,10))
for i in range(num_train):
  y_train1[i] = labels2[i][0]
  y_train2[i] = labels2[i][1]
  y_train3[i] = labels2[i][2]
  y_train4[i] = labels2[i][3]
  y_train5[i] = labels2[i][4]
  y_train6[i] = labels2[i][5]
  y_train7[i] = labels2[i][6]







######################### read in x images ######################
os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cp_train\\train')
# create training set
x_train =[]
for i in range(num_train):
  s = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE) 
  x_train.append(s)


x_test  =[]
for j in range(50):    
  s = cv2.imread("%s.png"%(i+(num_train+1)),cv2.IMREAD_GRAYSCALE) 
  x_test.append(s)
  
  
  ##normalize data


x_test = np.asarray(x_train)
x_test = x_train.astype('float32')
x_test = x_test / 255.0




######### y_test #######
y_test1=np.empty(shape = (num_test,10))
y_test2=np.empty(shape = (num_test,10))
y_test3=np.empty(shape = (num_test,10))
y_test4=np.empty(shape = (num_test,10))
y_test5=np.empty(shape = (num_test,10))
y_test6=np.empty(shape = (num_test,10))
y_test7=np.empty(shape = (num_test,10))
for i in range(num_test):
  y_test1[i] = labels2[i+num_train][0]
  y_test2[i] = labels2[i+num_train][1]
  y_test3[i] = labels2[i+num_train][2]
  y_test4[i] = labels2[i+num_train][3]
  y_test5[i] = labels2[i+num_train][4]
  y_test6[i] = labels2[i+num_train][5]
  y_test7[i] = labels2[i+num_train][6]







################################ CNN 1 ############################




inputs = tf.keras.layers.Input(shape=(423,625,1), name='input') 
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




# Prediction layers
c0 = tf.keras.layers.Dense(10, activation='softmax')(x)
c1 = tf.keras.layers.Dense(10, activation='softmax')(x)
c2 = tf.keras.layers.Dense(10, activation='softmax')(x)
c3 = tf.keras.layers.Dense(10, activation='softmax')(x)
c4 = tf.keras.layers.Dense(10, activation='softmax')(x)
c5 = tf.keras.layers.Dense(10, activation='softmax')(x)
c6 = tf.keras.layers.Dense(10, activation='softmax')(x)

# Defining the model
model = tf.keras.Model(inputs=inputs,outputs=[c0,c1,c2,c3,c4,c5,c6])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model

model.fit(x=x_train,y = [y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7]  , batch_size=1, epochs=1) 