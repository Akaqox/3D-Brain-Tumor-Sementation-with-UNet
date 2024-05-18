#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:25:44 2024

@author: solauphoenix
"""

import os
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import nibabel as nib                                                     
import itk                                                                
import itkwidgets
import cv2
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


IMG_START_AT=56
IMG_SIZE=128
TRAIN_DATASET_PATH = '../base_dir/train_ds'
keras = tf.compat.v1.keras
Sequence = keras.utils.Sequence

VOLUME_SLICES = 128 
VOLUME_START_AT = 22

class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for 3D convolutional U-Net brain segmentation'''

    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE, IMG_SIZE),
                 batch_size=1, n_channels=2, shuffle=True):
        '''Initialization'''
        super().__init__()
        self.dim = dim  # Resized volume dimensions (128 x 128 x 128)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels  # Number of channels (Flair)
        self.shuffle = shuffle
        self.on_epoch_end()  # Updates indexes after each epoch

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(len(self.list_IDs) / self.batch_size)  # Consider full volume for batches

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        '''Generates data containing batch_size samples'''
        # Generate data
        scaler = MinMaxScaler()
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)
            
            data_path = os.path.join(case_path, f'{i}_flair.nii.gz')
            t2 = nib.load(data_path).get_fdata()
            t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
            
            # data_path = os.path.join(case_path, f'{i}_t1ce.nii.gz')
            # t1ce = nib.load(data_path).get_fdata()
            # t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
            
            data_path = os.path.join(case_path, f'{i}_seg.nii.gz')
            seg = nib.load(data_path).get_fdata()
            # One-hot encode masks (assuming 4 classes)
            seg = seg.astype(np.uint8)
            seg[seg == 4] = 3 
            # Load full volumes (assuming 3D data)
   #         X = np.stack((
            X = np.expand_dims(np.expand_dims(cv2.resize(t2[IMG_START_AT:IMG_START_AT+IMG_SIZE, IMG_START_AT:IMG_START_AT+IMG_SIZE, VOLUME_START_AT:VOLUME_START_AT + VOLUME_SLICES], self.dim[:2]), axis=0), axis=-1)#,
                # np.expand_dims(cv2.resize(t1ce[IMG_START_AT:IMG_START_AT+IMG_SIZE, IMG_START_AT:IMG_START_AT+IMG_SIZE, VOLUME_START_AT:VOLUME_START_AT + VOLUME_SLICES], self.dim[:2]), axis=0)
    #            ), axis=-1)
                # Resize the one-hot encoded mask to match input dimensions
            y = np.expand_dims(seg[IMG_START_AT:IMG_START_AT+IMG_SIZE, IMG_START_AT:IMG_START_AT+IMG_SIZE, VOLUME_START_AT:VOLUME_START_AT + VOLUME_SLICES], axis=0)           
            y  = to_categorical(y, num_classes=4)
            
        # Normalize the data
        X = X / np.max(X)
        
        # Adapt for 3D mask if necessary (e.g., one-hot encoding)
        return X, y