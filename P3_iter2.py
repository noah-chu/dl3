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

# read in JSON file 
dir = '/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN(2)/train'
os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN(2)/train')
f = open('digitStruct.json',)
data = json.load(f)


## Only Do Once ##
# os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN\\train')
# dataset_url = "http://ballings.co/SVHN.zip"
# data_dir = tf.keras.utils.get_file(origin=dataset_url,
#                                    fname='photos',
#                                    untar=True)

## check images ##
data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN(2)')
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file
plt.imshow(PIL.Image.open(str(image[0])))
plt.show()



### grayscale and store ###
os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_grayscaled\\train')
for i in range(image_count):
  img = cv2.imread(str(image[i]))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #cv2.imshow("result", img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  # save a image using extension
  cv2.imwrite("%s.png"%(i+1), img)


### find max shape of grayscaled images ###

data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN')
image_count = len(list(data_dir.glob('*/*.png')))
image_gray = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470)

### grayscale and store ###
img = cv2.imread(str(image[0]))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
  
# save a image using extension
os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN\\grayscale')
cv2.imwrite('1.png', img)


### pad grayscaled images ###

  #evaluate max dimensions over all pictures 
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_grayscaled_(all)')
image_count = len(list(data_dir.glob('*/*.png')))
image_gray = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)


"""
:param images: sequence of images
:return: list of images padded so that all images have same width and height (max width and height are used)
"""

width_max = 0
height_max = 0
for i in range(image_count):
    img = cv2.imread(str(image_gray[i]))
    h, w = img.shape[:2]
    width_max = max(width_max, w)
    height_max = max(height_max, h)
    


#pad images, but store in separate train/test files

data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_grayscaled_train')
image_count = len(list(data_dir.glob('*/*.png')))
image_gray = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)



os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train\\train')
images_padded = []
for i in range(image_count):
    img = cv2.imread(str(image_gray[i]))
    h, w = img.shape[:2]
    diff_vert = height_max - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top
    diff_hori = width_max - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert img_padded.shape[:2] == (height_max, width_max)
    cv2.imwrite("%s.png"%(i+1), img_padded)










yt = np_utils.to_categorical(y_train)












############### BEGIN CNN Process ################
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train')
image_count = len(list(data_dir.glob('*/*.png')))
I = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file (46470) (test = 0 : 13067) (train = 13068 : end)
s = cv2.imread(str(I[0]))


# create training set
x_train =[]
y_train=[]
for i in range(20):
  if i < 10:
    s = cv2.imread(str(I[i])) 
    x_train.append(s)
  else:
    s = cv2.imread(str(I[i])) 
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




x_test =[]
y_test=[]
for i in range(20):
  if i < 10:
    s = cv2.imread(str(I[i])) 
    x_test.append(s)
  else:
    s = cv2.imread(str(I[i])) 
    y_test.append(s)







x_test = np.asarray(x_test)
x_test = x_train.astype('float32')
y_test = np.asarray(y_test)
y_test = y_test.astype('float32')
#x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = y_test / 255.0
#x_test = x_test / 255.0










  ###### CNN Layers



inputs = tf.keras.layers.Input(shape=(516,1083,3), name='input') 
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









# inputs = tf.keras.layers.Input(shape=(None,516,1083,3), name='input') 
# #Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
# x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(inputs)
# #MaxPooling2D: pool_size is window size over which to take the max
# x = tf.keras.layers.MaxPooling2D(pool_size = 6, strides = 2, padding = "valid")(x)
# x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
# x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
# x = tf.keras.layers.MaxPooling2D(pool_size = 6, strides = 2, padding = "valid")(x)
# x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
# x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
# x = tf.keras.layers.MaxPooling2D(pool_size = 6, strides = 2, padding = "valid")(x)
# #dense layers expect 1D array of features for each instance so we need to flatten.
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(128, activation = 'relu')(x)
# x = tf.keras.layers.Dense(64, activation = 'relu')(x)
# yhat = tf.keras.layers.Dense(6, activation = 'softmax')(x)






















  #### compile model
model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))


history = model.fit(x=x_train,y=yt, batch_size=1, epochs=1)

y_train.shape


y_train[0].shape







































