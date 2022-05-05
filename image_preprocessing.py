from curses.panel import bottom_panel
from importlib.resources import path
from turtle import width
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


############# JSON Info ###############

try:
       os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN_Padded_Train/train')
except FileNotFoundError:
       os.chdir("G:\\My Drive\\MSBA\\Spring\\Deep Learning\\GA3\\SVHN_Padded_train\\train")
jdata = pd.read_json(r'digitStruct.json')
jdata.head()

jdata.iloc[0,0]
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




top ={}
width={}
height={}
left={}
for row in range(jdata.shape[0]): #grab each row of the jdata frame
  rowlist_top = np.array([])
  #rowlist_bottom = np.array([])
  rowlist_width = np.array([])
  rowlist_height = np.array([])
  rowlist_left = np.array([])

  for box in jdata.iloc[row,0]: #inspect each number of the picture
    #append it to an array
    rowlist_top = np.append(rowlist_top, box['top'])
    #rowlist_bottom = np.append(rowlist_bottom, box['bottom'])
    rowlist_width = np.append(rowlist_width, box['width'])
    rowlist_height = np.append(rowlist_height, box['height'])
    rowlist_left = np.append(rowlist_left, box['left'])
  #add the labels of this entry to the dictionary
  top[row] = rowlist_top
  #bottom[row] = rowlist_bottom
  width[row] = rowlist_width
  height[row] = rowlist_height
  left[row] = rowlist_left




top[0]








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


























## check images ##
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_train')
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file
plt.imshow(PIL.Image.open(str(image[1])))
plt.show()

img = cv2.imread(str(image[3]))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(img)
plt.show()






###### crop images ##########




data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_train') #path to original pics
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file


img = cv2.imread(str(image[1]))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(img)
plt.show()



img = cv2.imread('2.png')
plt.imshow(img)
plt.show()

os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cropped_train\\train') #where new pics are stored

#int(min(left[i])):int(min(left[i])+max(width[i])), int(min(top[i])):int(max(top[i])+ max(height[i]))


i = 1


####### crop train #########

for i in range(image_count):
    #roi=cv2.selectROI(img)
    os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_train\\train') 
    img = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE)
    roi_cropped=img[int(min(top[i])):int(max(top[i])+max(height[i])), int(min(left[i])):int(max(left[i])+ max(width[i]))]
    os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cropped_train\\train')
    try:
        cv2.imwrite("%s.png"%(i+1), roi_cropped)
    except:
        pass















####### crop test #########




try:
       os.chdir('/Users/Harrison Eller/OneDrive/Desktop/MSBA/Spring 2022/BZAN 554 - Deep Learning/SVHN_test/test')
except FileNotFoundError:
       os.chdir("G:\\My Drive\\MSBA\\Spring\\Deep Learning\\GA3\\SVHN_Padded_train\\train")
jdata = pd.read_json(r'digitStruct.json')
jdata.head()

jdata.iloc[0,0]
################  LABELS ########################
# #initialize empty dictionary 
# labels = {}
# for row in range(jdata.shape[0]): #grab each row of the jdata frame
#   rowlist = np.array([])
#   for box in jdata.iloc[row,0]: #inspect each number of the picture
#     #append it to an array
#     rowlist = np.append(rowlist, box['label'])
#   #add the labels of this entry to the dictionary
#   labels[row] = rowlist
# labels[0]




top ={}
width={}
height={}
left={}
for row in range(jdata.shape[0]): #grab each row of the jdata frame
  rowlist_top = np.array([])
  #rowlist_bottom = np.array([])
  rowlist_width = np.array([])
  rowlist_height = np.array([])
  rowlist_left = np.array([])

  for box in jdata.iloc[row,0]: #inspect each number of the picture
    #append it to an array
    rowlist_top = np.append(rowlist_top, box['top'])
    #rowlist_bottom = np.append(rowlist_bottom, box['bottom'])
    rowlist_width = np.append(rowlist_width, box['width'])
    rowlist_height = np.append(rowlist_height, box['height'])
    rowlist_left = np.append(rowlist_left, box['left'])
  #add the labels of this entry to the dictionary
  top[row] = rowlist_top
  #bottom[row] = rowlist_bottom
  width[row] = rowlist_width
  height[row] = rowlist_height
  left[row] = rowlist_left



for i in range(image_count):
    #roi=cv2.selectROI(img)
    os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_test\\test') 
    img = cv2.imread("%s.png"%(i+1),cv2.IMREAD_GRAYSCALE)
    roi_cropped=img[int(min(top[i])):int(max(top[i])+max(height[i])), int(min(left[i])):int(max(left[i])+ max(width[i]))]
    os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cropped_test\\test')
    try:
        cv2.imwrite("%s.png"%(i+1), roi_cropped)
    except:
        pass





plt.imshow(img)
plt.show()


































#pad images, grayscale, then store in separate train/test files
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cropped_test') #path to original pics
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file

height_max = 423
width_max = 625
"""
:param images: sequence of images
:return: list of images padded so that all images have same width and height (max width and height are used)
"""

width_max = 0
height_max = 0
for i in range(image_count):
    img = cv2.imread(str(image[i]), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    width_max = max(width_max, w)
    height_max = max(height_max, h)
    




data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cropped_test') #path to original pics
image_count = len(list(data_dir.glob('*/*.png')))
image = list(data_dir.glob('*/*.png'))
print(image_count) #number of images in file


os.chdir('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_cp_test\\test') #where new pics are stored
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
    cv2.imwrite("%s.png"%(i+1), img_padded)

