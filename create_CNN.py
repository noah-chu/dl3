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



<<<<<<< HEAD





##### labels ##########
#dir = '/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN(2)/train'
dir = '/Users/harri/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN(2)/train'
#os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN(2)/train')
os.chdir('/Users/harri/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN_Padded_train/train')
f = open('digitStruct.json',)
data = json.load(f)

#load the json file
try:
       os.chdir('/Users/harri/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN/train')
       #os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN/train')
=======
#load the json file
try:
       os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN_Padded_Train/train')
>>>>>>> 8ca457b431e3405a1bc34e98de54566cd16c514f
except FileNotFoundError:
       os.chdir("G:\\My Drive\\MSBA\\Spring\\Deep Learning\\GA3\\SVHN_Padded_train\\train")
jdata = pd.read_json(r'digitStruct.json')
jdata.head()

<<<<<<< HEAD

=======
################  LABELS ########################
>>>>>>> 8ca457b431e3405a1bc34e98de54566cd16c514f
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

<<<<<<< HEAD

=======
>>>>>>> 8ca457b431e3405a1bc34e98de54566cd16c514f
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









<<<<<<< HEAD
=======
# Converting labels to One-hot representations of shape (set_size, digits, classes)
possible_classes = 11

def convert_labels(labels):
    
    # As per Keras conventions, the multiple labels need to be of the form [array_digit1,...5]
    # Each digit array will be of shape (60000,11)
        
    # Declare output ndarrays
    # 5 for digits, 11 for possible classes  
    dig0_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig1_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig2_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig3_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig4_arr = np.ndarray(shape=(len(labels),possible_classes))
    
    for index,label in enumerate(labels):
        
        # Using np_utils from keras to OHE the labels in the image
        dig0_arr[index,:] = np_utils.to_categorical(label[0],possible_classes)
        dig1_arr[index,:] = np_utils.to_categorical(label[1],possible_classes)
        dig2_arr[index,:] = np_utils.to_categorical(label[2],possible_classes)
        dig3_arr[index,:] = np_utils.to_categorical(label[3],possible_classes)
        dig4_arr[index,:] = np_utils.to_categorical(label[4],possible_classes)
        
    return [dig0_arr,dig1_arr,dig2_arr,dig3_arr,dig4_arr]




>>>>>>> 8ca457b431e3405a1bc34e98de54566cd16c514f













#data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')#laptop
<<<<<<< HEAD
data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')#desktop
image_count = len(list(data_dir.glob('*/*.png')))
I = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
s = cv2.imread(str(I[1]),cv2.IMREAD_GRAYSCALE)
s.shape
46470-13067
=======
# data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cp_train')#desktop
# image_count = len(list(data_dir.glob('*/*.png')))
# I = list(data_dir.glob('*/*.png'))
# print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
# s = cv2.imread(str(I[1]),cv2.IMREAD_GRAYSCALE)
# s.shape

>>>>>>> 8ca457b431e3405a1bc34e98de54566cd16c514f

plt.imshow(s)
plt.show()


os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cp_train\\train')
# create training set
x_train =[]
y_train=[]
for i in range(20):
  if i < 10:
    s = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE) 
    x_train.append(s)
  else:
    s = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE)
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



#data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_test')#laptop
# data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_test')#desktop
# image_count = len(list(data_dir.glob('*/*.png')))
# I = list(data_dir.glob('*/*.png'))
# print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
# s = cv2.imread(str(I[0]))
# s.shape



os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cp_test\\test')
x_test =[]
y_test=[]
for i in range(20):
  if i < 10:
    s = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE) 
    x_test.append(s)
  else:
    s = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE) 
    y_test.append(s)


x_test = np.asarray(x_test)
x_test = x_train.astype('float32')
y_test = np.asarray(y_test)
y_test = y_test.astype('float32')
#x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = y_test / 255.0
#x_test = x_test / 255.0



# READ IN EVERYTHING UP TO HERE # 


# Do we OHE y? and is it done as seen below? #


y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)



















<<<<<<< HEAD
=======
y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)

>>>>>>> 8ca457b431e3405a1bc34e98de54566cd16c514f



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
            x = tf.keras.layers.Dense(1083,activation = 'softmax')(x)
            
         
            if regress:
                x = tf.keras.Dense(1, activation="linear")(x)
	# construct the CNN
            model =tf.keras.Model(inputs, x)
	# return the CNN
            return model
        
tf.keras.backend.clear_session()
model = create_cnn(625,423,1)

x_train.shape

model.summary()
model.compile(loss='mean_absolute_percentage_error',optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
history = model.fit(x=x_train,y=y_train_ohe, validation_data=(x_test,y_test), batch_size=32, epochs=1)



















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
yhat = tf.keras.layers.Dense(10, activation = 'softmax')(x)





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
model.fit(x=x_train,y=y_train, batch_size=10, epochs=1) 


























###################### CNN Try 5 ################################




list(labels[0])



      

possible_classes = 11

def convert_labels(labels):
    
    # As per Keras conventions, the multiple labels need to be of the form [array_digit1,...5]
    # Each digit array will be of shape (60000,11)
        
    # Declare output ndarrays
    # 5 for digits, 11 for possible classes  
    dig0_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig1_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig2_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig3_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig4_arr = np.ndarray(shape=(len(labels),possible_classes))
    
    for index,label in enumerate(labels):
        
        # Using np_utils from keras to OHE the labels in the image
        dig0_arr[index,:] = np_utils.to_categorical(label[0],possible_classes)
        dig1_arr[index,:] = np_utils.to_categorical(label[1],possible_classes)
        dig2_arr[index,:] = np_utils.to_categorical(label[2],possible_classes)
        dig3_arr[index,:] = np_utils.to_categorical(label[3],possible_classes)
        dig4_arr[index,:] = np_utils.to_categorical(label[4],possible_classes)
        
    return [dig0_arr,dig1_arr,dig2_arr,dig3_arr,dig4_arr]









x_labs = convert_labels(labels[0])




train_labels = convert_labels(x_train)
test_labels = convert_labels(test_labels)
valid_labels = convert_labels()
















batch_size = 32
nb_classes = 11
nb_epoch = 24



# number of convulation filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# defining the input
inputs = tf.keras.Input(shape=(516, 1083, 1))

# Model taken from keras example.
cov = tf.keras.layers.Conv2D(nb_filters,kernel_size=(kernel_size[0],kernel_size[1]),padding='same', use_bias=False)(inputs)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.Conv2D(nb_filters,kernel_size=(kernel_size[0],kernel_size[1]),padding='same', use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(cov)
cov = tf.keras.layers.Dropout(0.3)(cov)

cov = tf.keras.layers.Conv2D((nb_filters * 2),kernel_size=(kernel_size[0],kernel_size[1]), use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.Conv2D((nb_filters * 2),kernel_size=(kernel_size[0],kernel_size[1]),padding='same', use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(cov)
cov = tf.keras.layers.Dropout(0.3)(cov)


cov = tf.keras.layers.Conv2D((nb_filters * 4),kernel_size=(kernel_size[0],kernel_size[1]), use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.Conv2D((nb_filters * 4),kernel_size=(kernel_size[0],kernel_size[1]),padding='same', use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(cov)
cov = tf.keras.layers.Dropout(0.3)(cov)

cov = tf.keras.layers.Conv2D((nb_filters * 8),kernel_size=(kernel_size[0],kernel_size[1]), use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.Conv2D((nb_filters * 8),kernel_size=(kernel_size[0],kernel_size[1]),padding='same', use_bias=False)(cov)
cov = tf.keras.layers.BatchNormalization()(cov)
cov = tf.keras.layers.Activation('relu')(cov)
cov = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(cov)
cov = tf.keras.layers.Dropout(0.3)(cov)

cov_out = tf.keras.layers.Flatten()(cov)


# Dense Layers
cov2 = tf.keras.layers.Dense(2056, activation='relu')(cov_out)
cov2 = tf.keras.layers.Dropout(0.3)(cov2)



# Prediction layers
c0 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)
c1 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)
c2 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)
c3 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)
c4 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)
c5 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)
c6 = tf.keras.layers.Dense(nb_classes, activation='softmax')(cov2)

# Defining the model
model = tf.keras.Model(inputs=inputs,outputs=[c0,c1,c2,c3,c4,c5,c6])
model.summary()


# Compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


    
# Fitting the model
model.fit(x_train,y_train_ohe,batch_size=batch_size,epochs=nb_epoch)