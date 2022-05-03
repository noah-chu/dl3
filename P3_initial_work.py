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


# check original image file
data_dir = pathlib.Path('C:\\Users\\Harrison Eller\\OneDrive\\Desktop\\MSBA\\Spring 2022\\BZAN 554 - Deep Learning\\SVHN_train')
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






























class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()