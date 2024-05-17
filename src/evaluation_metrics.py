#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:34:07 2024

@author: solauphoenix
"""

epsilon = 1e-5
smooth = 1

import tensorflow as tf
from tensorflow.keras import backend as K

# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, epsilon=0.00001):
    y_true = K.cast(y_true, 'float32')
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return tf.reduce_mean((dice_numerator)/(dice_denominator))

    # inspired by https://github.com/keras-team/keras/issues/9395
    
def dice_coef_necrotic(y_true, y_pred, epsilon=0.00001):
    y_true = K.cast(y_true, 'float32')
    intersection = K.sum(K.abs(y_true[0,:,:,:,1] * y_pred[0,:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[0,:,:,:,1])) + K.sum(K.square(y_pred[0,:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=0.00001):
    y_true = K.cast(y_true, 'float32')
    intersection = K.sum(K.abs(y_true[0,:,:,:,2] * y_pred[0,:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[0,:,:,:,2])) + K.sum(K.square(y_pred[0,:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=0.00001):
    y_true = K.cast(y_true, 'float32')
    intersection = K.sum(K.abs(y_true[0,:,:,:,3] * y_pred[0,:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[0,:,:,:,3])) + K.sum(K.square(y_pred[0,:,:,:,3])) + epsilon)


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

#dice-coefs and tversky are combined with some coef
def combinational_loss(y_true, y_pred):
    return (1 - tversky(y_true,y_pred)) + (1 - dice_coef_necrotic(y_true, y_pred))*0.3 + (1 - dice_coef_enhancing(y_true, y_pred))*0.2 +(1 - dice_coef(y_true, y_pred))*0.1
