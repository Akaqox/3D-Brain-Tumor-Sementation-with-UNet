#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:34:09 2024

@author: solauphoenix
"""

import keras
import matplotlib.pyplot as plt
import matplotlib
import random
import time
import evaluation_metrics as em
import numpy as np

IMG_START_AT=56
IMG_SIZE=128
TRAIN_DATASET_PATH = '../base_dir/train_ds'
VOLUME_SLICES = 128 
VOLUME_START_AT = 22

# Plot (and save) the graphs of loss and metric values

# def evaluation():
    
#     num_test_samples = 10  # Adjust this based on your desired number of test items
#     X_test = []
#     y_test = []
    
#     for i in range(num_test_samples):
#         X_item, y_item = test_generator.__getitem__(i)
#         print(X_item.shape)
#         print(X_item.shape)
#         X_test.append(X_item)
#         y_test.append(y_item)
    
#     X_test = np.array(X_test)  # Convert to NumPy array for efficient processing
#     y_test = np.array(y_test)
    
#     score = model.evaluate(   
#     x=X_test,
#     y=y_test,
#     batch_size=16,
#     verbose="auto",)
    
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
#     print('Test precision:', score[2])
#     print('Test recall:', score[3])
    



def plot_performance_curve(training_result, metric, metric_label):

    train_perf = training_result.history[str(metric)]
    validation_perf = training_result.history['val_' + str(metric)]

    # Filter out epochs with all zeros
    valid_epochs = [i for i in range(len(train_perf)) if train_perf[i] != 0 and validation_perf[i] != 0]

    train_perf = [train_perf[i] for i in valid_epochs]
    validation_perf = [validation_perf[i] for i in valid_epochs]
    plt.plot(train_perf, label=metric_label)
    plt.plot(validation_perf, label='val_' + str(metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric_label)
    plt.legend(loc='lower right')



def predict_ten(test_generator):
    #for 10 sample print results
    current_time = time.time()
    
    
    custom_objects ={'combinational_loss' :  em.combinational_loss,
                     'dice_coef': em.dice_coef, 
                     'dice_coef_edema': em.dice_coef_edema, 
                     'dice_coef_enhancing': em.dice_coef_enhancing,
                     'dice_coef_necrotic': em.dice_coef_necrotic}
    
    
    model1 = keras.models.load_model("model_Unet_2mod.keras", custom_objects=custom_objects)
    
    print("Evaluate on test data")
    for i in range(2):
        X, y_real = test_generator.__getitem__(i)
        
        
        print("y predicted shape:", y_real.shape)
        print("X shape:", X.shape)
        
        
        data = model1.predict(X)
        data_real=y_real
        
        z_slice = 64
        
        # data_argmax = np.argmax(data, axis=-1)  # Find the class with the highest probability for each pixel

        # # Resize the class labels back to the original image dimensions (if necessary)
        # data = np.zeros_like(data, dtype=np.uint8)
        # print(data.shape)
        # data[: , :, :, :] = data_argmax
       
                
        
        # Create a single figure with two subplots
        fig, (ax_input, ax_predicted, ax_real) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
        
    
        # Plot input data
        ax_input.imshow(X[0, :, :, z_slice,0])
        ax_input.set_title('Input')
        
        
        for i, color in enumerate([ 'jet', 'jet', 'jet', 'jet']):
            channel = data[0, :, :, z_slice, i]  # Access channel data (assuming starts from index 1)
            ax_predicted.imshow(channel, cmap=color, alpha=0.6)
            #ax_predicted.set_title(channel_title, fontsize=12)
            fig.suptitle(f"Z-Slice {z_slice} (4 Channels)", fontsize=14)

            # Add a main title or adjust spacing for better visualization
        for i, color in enumerate(['jet', 'jet', 'jet', 'jet']):
            
            channel = data_real[0, :, :, z_slice, i]  # Access channel data (assuming starts from index 1)
            ax_real.imshow(channel, cmap=color, alpha=0.6)
            #ax_predicted.set_title(channel_title, fontsize=12)


            # Add a main title or adjust spacing for better visualization
        fig.suptitle(f"Z-Slice {z_slice} (4 Channels)", fontsize=14)
        plt.tight_layout()
        ax_real.set_title('Ground Truth')    
        
        plt.subplots_adjust(wspace=0)
        plt.savefig('../Plot/predict/ ' + str(i) +' '+ str(current_time) + ' .png')
        plt.show()

