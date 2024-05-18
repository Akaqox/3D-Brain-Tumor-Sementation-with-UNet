from keras.models import Model
from keras import activations
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Activation, Lambda

kernel_initializer = 'he_uniform' 
    
def Unet_3d(input_img, n_filters = 4, dropout = 0.2, batch_norm = True, num_classes = 4):
    
    #Downsampling Convolutions
    con1 = conv_block(input_img, n_filters*8, 3, True, 0.2)
    pool1 = MaxPooling3D((2, 2, 2))(con1)
    #X = layers.add([input_mat, X])
    
    con2 = conv_block(pool1, n_filters*16, 3, True, 0.2)
    pool2 = MaxPooling3D((2, 2, 2))(con2)
    
    con3 = conv_block(pool2, n_filters*32, 3, True, 0.2)
    pool3 = MaxPooling3D((2, 2, 2))(con3)
    con4 = conv_block(pool3, n_filters*64, 3, True, 0.2)
    pool4 = MaxPooling3D((2, 2, 2))(con4)
    
    
    #Center of model
    con5 = conv_block(pool4, n_filters*128, 3, True, 0.2)
    
    
    #Umpsampling Expensive Convolutions
    
    up6 = Conv3DTranspose(n_filters*64, (2, 2, 2), strides=(2, 2, 2), padding='same')(con5)
    #connected to con4
    up6 = concatenate([up6, con4])
    con6 = conv_block(up6, n_filters*64, 3, True, 0.2)
    up7 = Conv3DTranspose(n_filters*32, (2, 2, 2), strides=(2, 2, 2), padding='same')(con6)
    #connected to con3
    up7 = concatenate([up7, con3])
    con7 = conv_block(up7, n_filters*32, 3, True, 0.2)
    
    up8 = Conv3DTranspose(n_filters*16, (2, 2, 2), strides=(2, 2, 2), padding='same')(con7)
    up8 = concatenate([up8, con2]) 
    #connected to con2
    con8 = conv_block(up8, n_filters*16, 3, True, 0.2)
    
    up9 = Conv3DTranspose(n_filters*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(con8)
    up9 = concatenate([up9, con1]) 
    #connected to con1
    con9 = conv_block(up9, n_filters*8, 3, True, 0.2)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(con9)
    
    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary
    return model


 #Downsampling Convolution Block Function
def conv_block(input_mat,num_filters,kernel_size,batch_norm,dropout_factor):
    X = Conv3D(num_filters,
               kernel_size=(kernel_size,kernel_size,kernel_size), 
               kernel_initializer=kernel_initializer, 
               activation='relu', 
               strides=(1,1,1),
               padding='same')(input_mat)
    
    if batch_norm:
      X = BatchNormalization()(X)
      
    X = Dropout(dropout_factor)(X)
    
    X = Conv3D(num_filters,
               kernel_size=(kernel_size,kernel_size,kernel_size), 
               kernel_initializer=kernel_initializer, 
               activation='relu', 
               strides=(1,1,1),
               padding='same')(X)
    
    if batch_norm:
      X = BatchNormalization()(X)

     
    
    return X
