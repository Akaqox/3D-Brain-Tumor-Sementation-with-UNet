#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:23:17 2024

@author: solauphoenix
"""



import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps  


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt


# ml libs
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
from tensorflow.keras.layers.experimental import preprocessing



data_path = "../archive/BraTS2021_Training_Data/"
scan_type = ("flair", "t1", "t1ce", "t2")


#reading scans of patients
def load_patient(p_number):
    for scan in scan_type:
        idx = str(p_number).zfill(5)
        path = f'{data_path}/BraTS2021_{idx}/BraTS2021_{idx}_{scan}.nii.gz'
        img  = nib.load(path)
    return 

load_patient(0)
# nii_data = img.get_fdata()
# nii_aff  = img.affine
# nii_hdr  = img.header
# print(nii_aff ,'\n',nii_hdr)
# print(nii_data.shape)


# display = nl.plotting.plot_img(img)
# nl.plotting.show()