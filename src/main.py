#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:49:42 2024

@author: solauphoenix
"""
import os
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
from sklearn import metrics as mtrc
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam

#other python codes
import data_generator as dg
import evaluation_metrics as em
#from Unet import simple_unet_model
import model as Unet
import evaluation_visualization as ev



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


train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 
    
#should be reviewed
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15) 

        
training_generator = dg.DataGenerator(train_ids)
valid_generator = dg.DataGenerator(val_ids)
test_generator = dg.DataGenerator(test_ids)

print(type(training_generator.__len__))
csv_logger = CSVLogger('training.log', separator=',', append=False)


callbacks = [
      # keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
      #                          patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=2, min_lr=0.000001, verbose=1),
      keras.callbacks.ModelCheckpoint(filepath = '../weights/model_.{epoch:02d}-{val_loss:.6f}.weights.h5',
                              verbose=1, save_best_only=True, save_weights_only = True),
      csv_logger
    ]

input_layer = Input((128, IMG_SIZE, IMG_SIZE, 2))
model = Unet.Unet_3d(input_img=input_layer, n_filters = 4, dropout = 0.2, batch_norm = True, num_classes = 4)



# input_layer = Input((128,IMG_SIZE, IMG_SIZE, 2))

model.compile(
    loss= em.combinational_loss, 
    optimizer=Adam(learning_rate=0.0005), 
    metrics = ['precision',
                em.dice_coef,
                em.dice_coef_edema ,
                em.dice_coef_enhancing,
                em.dice_coef_necrotic])




#100 step size 1 epoch first try
#300 step size 10 eoch second try
#600 step size 20 epoch third try
#200 step size 60 epoch fourth try
# len(train_ids) 10 epoch the best
# len(train_ids) 20 epoch the len(train_ids) led some metric problems so we
#have to reduce batch_size
#170 step size 30 epoch 0.001 step size (I saw that metric problem was due to running out the data)


EPOCH = 40
steps_per_epoch = 851

history =  model.fit(training_generator,
                      epochs=EPOCH,
                      # steps_per_epoch=len(train_ids)
                      steps_per_epoch=steps_per_epoch,
                      callbacks= callbacks,
                      validation_data = valid_generator)  

model.save("model_Unet_2mod.keras")
# Evaluate your model's performance
#will be added

custom_objects ={'combinational_loss' :  em.combinational_loss,
                 'dice_coef': em.dice_coef, 
                 'dice_coef_edema': em.dice_coef_edema, 
                 'dice_coef_enhancing': em.dice_coef_enhancing,
                 'dice_coef_necrotic': em.dice_coef_necrotic}
   
model1 = keras.models.load_model("model_Unet_2mod.keras", custom_objects=custom_objects)

model1.evaluate(test_generator)

ev.predict_ten(test_generator)

ev.plot_performance_curve(history, 'loss', 'loss')
ev.plt.savefig('../Plot/loss_curve ' + str(EPOCH) +' '+ str(steps_per_epoch) + '.png', dpi=300)
plt.clf()

ev.plot_performance_curve(history, 'dice_coef_edema', 'dice_coef_edema')
ev.plot_performance_curve(history, 'dice_coef_enhancing', 'dice_coef_enhancing')
ev.plot_performance_curve(history, 'dice_coef_edema', 'dice_coef_edema')
ev.plot_performance_curve(history, 'dice_coef_necrotic', 'dice_coef_necrotic')
plt.savefig('../Plot/metrics ' + str(EPOCH) +' '+ str(steps_per_epoch) + ' .png', dpi=300)
