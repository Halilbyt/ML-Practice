import numpy as np 
import matplotlib.pyplot as plt

import cv2 as cv
#import PIL as pl

import requests
import time

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers  import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
##############################################################################################################################################

def load_img(path):
    '''
    reading image , basic implamentation
    inputs =  image name that located in the folder (string)
    '''
    try:
        img = cv.imread(f'C:\\Users\\D4rkS\\Desktop\\Code_Lab\\Python_Basics\\IP_DSP_DNN_ANN\\{path}')
        #img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # => for matplotlib
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        return img, img_gray
    except:
        raise ValueError('inputs not correct formath')
def show_img(img,fig_nums = None):
    cv.imshow('Figure',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# loading image;

img,img_gray = load_img('land.jpg')

# Image Processing Basic
# flip img, drawing basic, put text on img
# Morphologic operations 
# Histogram and equalization
# Threshold and contours
# Cam applications
# Face and object detections

ret,thr = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
contours, hiararchy = cv.findContours(thr,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#rint(type(contours),type(hiararchy))

for i in contours:
    (x,y,w,h) = cv.boundingRect(i)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),7)

# face detections

face_cascade = cv.CascadeClassifier(r'C:\Users\D4rkS\Desktop\Code_Lab\Python_AI\Computer-Vision-with-Python\DATA\haarcascades\haarcascade_frontalface_alt_tree.xml')
def face_detect(frm):
    frm_copy = frm.copy()
    face_det =  face_cascade.detectMultiScale(frm_copy,scaleFactor = 1.2, minNeighbors = 5)
    for (x,y,w,h) in face_det:
        cv.rectangle(frm_copy,(x,y),(x+w,y+h),(0,255,0),7)
    return frm_copy

cap = cv.VideoCapture(0)

# HSV calc
class HSVbar:
    
    def __init__(self,w=None,h=None):
        self.w = w  # window width
        self.h = h # window height
                    
    def nothing(self):
        pass
    def cereate_bar(self,ret = False):
               
        if ret == True:
            cv.namedWindow('HSV BAR')
            cv.resizeWindow('HSV BAR',(self.w,self.h))

            cv.createTrackbar('Lower - H', 'HSV BAR', 0, 180, self.nothing)
            cv.createTrackbar('Lower - S', 'HSV BAR', 0, 255, self.nothing)
            cv.createTrackbar('Lower - V', 'HSV BAR', 0, 255, self.nothing)

            cv.createTrackbar('Upper - H', 'HSV BAR', 0, 180, self.nothing)
            cv.createTrackbar('Upper - S', 'HSV BAR', 0, 255, self.nothing)
            cv.createTrackbar('Upper - V', 'HSV BAR', 0, 255, self.nothing)

            cv.setTrackbarPos('Upper - H', 'HSV BAR',180)
            cv.setTrackbarPos('Upper - S', 'HSV BAR',255)
            cv.setTrackbarPos('Upper - V', 'HSV BAR',255)

bar = HSVbar(300,300)
bar.cereate_bar()

# finding contours on cap

def tracking(frame):
       
    frame_copy = frame.copy()
    ret,th = cv.threshold(frame_copy,100,255,cv.THRESH_BINARY)
    contrs, hier = cv.findContours(th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for i in contrs:
        (x,y,w,h) = cv.boundingRect(i)
        cv.rectangle(frame_copy,(x,y),(x+w,y+h),(255,0,0),7)
    return frame_copy


def showCam():
    while True:
        ret,frame = cap.read()
        frame = cv.flip(frame,1)
        frame_hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

        lower_h = cv.getTrackbarPos('Lower - H','HSV BAR')
        lower_s = cv.getTrackbarPos('Lower - S','HSV BAR')
        lower_v  = cv.getTrackbarPos('Lower - V','HSV BAR')
        lower_color = np.array([lower_h,lower_s,lower_v])

        upper_h = cv.getTrackbarPos('Upper - H','HSV BAR')
        upper_s = cv.getTrackbarPos('Upper - S','HSV BAR')
        upper_v = cv.getTrackbarPos('Upper - V','HSV BAR')
        upper_color = np.array([upper_h,upper_s,upper_v])

        hsv_mask = cv.inRange(frame_hsv,lower_color,upper_color)
        res2 = tracking(hsv_mask)
        
        if ret!=True:
            print('There is no captions')
            break
        res = face_detect(frame)
        cv.imshow('Face Detection',res)
        cv.imshow('HSV Color',hsv_mask)
        cv.imshow('Contour HSV',res2)
        k = cv.waitKey(20) & 0xFF
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()


    # CNN Model and evaluate

model = Sequential()
    
model.add(Conv2D(64,(3,3),activation = 'relu',input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
    
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

# parameters of Image Data Generator
''''
(class) ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
 samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=0.000001, rotation_range=0, width_shift_range=0, 
 height_shift_range=0, brightness_range=None, shear_range=0, zoom_range=0, channel_shift_range=0, fill_mode='nearest', cval=0,
 horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0, dtype=None)
'''
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_data_gen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    zoom_range = 0.2,
                                    shear_range = 0.2,
                                    fill_mode = 'nearest',
                                    horizontal_flip = True)

test_data_gen = ImageDataGenerator(rescale = 1./255)                                    
validation_data_gen = ImageDataGenerator(rescale = 1./255)

path_train = r'C:\Users\D4rkS\Desktop\Code_Lab\Python_AI\Computer-Vision-with-Python\DATA\CATS_DOGS\train' 
path_test = r'C:\Users\D4rkS\Desktop\Code_Lab\Python_AI\Computer-Vision-with-Python\DATA\CATS_DOGS\test'
path_val = r'C:\Users\D4rkS\Desktop\Code_Lab\Python_AI\Computer-Vision-with-Python\DATA\CATS_DOGS\validation' 

train_data = train_data_gen.flow_from_directory(directory = path_train,
                                                target_size = (150,150),
                                                batch_size = 32,
                                                class_mode = 'binary')
test_data = test_data_gen.flow_from_directory(directory = path_test,
                                                target_size = (150,150),
                                                batch_size = 32,
                                                class_mode = 'binary')
valid_data = validation_data_gen.flow_from_directory(directory = path_val,
                                                target_size = (150,150),
                                                batch_size = 32,
                                                class_mode = 'binary') 

# showing some img

img_bag = train_data.next(); img = img_bag[0]
#print(img)

#cv.imshow('img of data generator',img[0])
earlyStop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 3, verbose = 1)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())
 
# Training on GPU

model.fit(train_data, epochs = 5, validation_data = valid_data, callbacks = earlyStop)