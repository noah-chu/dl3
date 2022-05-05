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


## check images ##
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_test')
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file
plt.imshow(PIL.Image.open(str(image[0])))
plt.show()

img = cv2.imread(str(image[0]))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(img)
plt.show()



#pad images, grayscale, then store in separate train/test files
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_grayscaled_(all)') #path to original pics
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file


"""
:param images: sequence of images
:return: list of images padded so that all images have same width and height (max width and height are used)
"""

width_max = 0
height_max = 0
for i in range(image_count):
    img = cv2.imread(str(image[i]))
    h, w = img.shape[:2]
    width_max = max(width_max, w)
    height_max = max(height_max, h)
    
width_max = 1083
height_max=516



data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_train') #path to original pics
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file


os.chdir('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_train') #where new pics are stored
images_padded = []
for i in range(2):
    img = cv2.imread(str(image[i]), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    diff_vert = height_max - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top
    diff_hori = width_max - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert img_padded.shape[:2] == (height_max, width_max)
    #cv2.imwrite("%s.png"%(i+1), img_padded)
    cv2.imwrite("%s.png"%(i+1), img_padded)
    





data_dir = pathlib.Path('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_test') #path to original pics
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file


os.chdir('C:\\Users\\harri\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_Padded_T') #where new pics are stored
images_padded = []
for i in range(image_count):
    img = cv2.imread(str(image[i]), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    diff_vert = height_max - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top
    diff_hori = width_max - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert img_padded.shape[:2] == (height_max, width_max)
    #cv2.imwrite("%s.png"%(i+1), img_padded)
    cv2.imwrite("%s.png"%(i+1), img_padded)
    

