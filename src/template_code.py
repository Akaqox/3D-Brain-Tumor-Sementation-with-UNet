#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:49:42 2024

@author: solauphoenix
"""

import numpy as np
import nibabel as nib                                                     
import itk                                                                
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
from skimage.util import montage 
from skimage.transform import rotate


import seaborn as sns
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

#other python codes
import data_generator as dg
import evaluation_metrics as em

image_path = "../base_dir/train_ds/BraTS2021_00495/BraTS2021_00495_t1.nii.gz"
image_obj = nib.load(image_path)
type(image_obj)

image_data = image_obj.get_fdata()
print(type(image_data))
image_data.shape

layer = 50
classes_dict = {
    'Normal': 0.,
    'Edema': 1.,
    'Non-enhancing tumor': 2.,
    'Enhancing tumor': 3. 
}

IMG_SIZE=128
TRAIN_DATASET_PATH = '../base_dir/train_ds'

VALIDATION_DATASET_PATH = '../base_dir/val_ds'
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5 
VOLUME_SLICES = 128 
VOLUME_START_AT = 22 # first slice of volume that we will include

import os

train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 
    
#should be reqwiewed
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 
print(val_ids)

keras = tf.compat.v1.keras
Sequence = keras.utils.Sequence


        
training_generator = dg.DataGenerator(train_ids)
valid_generator = dg.DataGenerator(val_ids)
test_generator = dg.DataGenerator(test_ids)


csv_logger = CSVLogger('training.log', separator=',', append=False)


callbacks = [keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
      keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.weights.h5',
                             verbose=1, save_best_only=True, save_weights_only = True),
      csv_logger
    ]


from Unet import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=128,IMG_WIDTH=128,IMG_DEPTH=128,IMG_CHANNELS=1,num_classes=4)





# input_layer = Input((128,IMG_SIZE, IMG_SIZE, 2))

# model1 = model.Vnet_3d(input_layer)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4),
                                                                                                                 em.dice_coef_edema ,em.dice_coef_enhancing] )

model.summary()

history =  model.fit(training_generator,
                      epochs=1,
                      # steps_per_epoch=len(train_ids)
                      steps_per_epoch=100,
                      callbacks= callbacks,
                      validation_data = valid_generator
                      )  
model.save("model_Unet_2mod.h5")

plot_model(model, 
           show_shapes = True,
           show_dtype=False,
           show_layer_names = True, 
           rankdir = 'TB', 
           expand_nested = False, 
           dpi = 70)


model1 = keras.models.load_model("model_vnet_2mod.h5")

print("Evaluate on test data")

X, y = test_generator.__getitem__(0)



print("X boyutu:", X.shape)
print("y boyutu:", y.shape)



import random


data = model1.predict(X)


z_slice = 64


z_slice_data = data[0, :, :, z_slice, 0]

# Kesitleri görselleştir
plt.figure(figsize=(15, 5))


plt.subplot(224)
plt.imshow(z_slice_data)
plt.title('Z Slice')

plt.show()

data=y


z_slice = 64


z_slice_data = data[0, :, :, z_slice, 0]

# Kesitleri görselleştir
plt.figure(figsize=(15, 5))

plt.subplot(224)
plt.imshow(z_slice_data)
plt.title('Z Slice')

plt.show()
