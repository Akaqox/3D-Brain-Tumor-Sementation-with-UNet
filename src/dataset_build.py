#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:10:55 2024

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



data_path = "../archive/BraTS2021_Training_Data/"
scan_type = ("flair", "t1", "t1ce", "t2")
base_dir="null"
train_dir="null"
val_dir="null"
test_dir="null"

#reading scans of patients
def load_patient(p_number):
    for scan in scan_type:
        idx = str(p_number).zfill(5)
        path = f'{data_path}/BraTS2021_{idx}/BraTS2021_{idx}_{scan}.nii.gz'
        img  = nib.load(path)
    return img
#Read the all diretories
def get_folder_names(directory_path):
  folder_names = []
  for item in os.listdir(directory_path):
    if os.path.isdir(os.path.join(directory_path, item)):
      folder_names.append(item)

  return folder_names
#Read the all diretories
def build_dataset():
    base_dir = '../base_dir'
    os.mkdir(base_dir)
    
    train_dir = os.path.join(base_dir, 'train_scan')
    os.mkdir(train_dir)
    
    val_dir = os.path.join(base_dir, 'val_scan')
    os.mkdir(val_dir)
    
    test_dir = os.path.join(base_dir, 'test_scan')
    os.mkdir(test_dir)
    print(train_dir)
# Example usage

folder_names = get_folder_names(data_path)

print("Folder names:")
for name in folder_names:
    
  print(name)
build_dataset()