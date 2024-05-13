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
    
    
    custom_objects ={'dice_coef': em.dice_coef, 
                     'dice_coef_edema': em.dice_coef_edema, 
                     'dice_coef_enhancing': em.dice_coef_enhancing,
                     'dice_coef_necrotic': em.dice_coef_necrotic}
    
    
    model1 = keras.models.load_model("model_Unet_2mod.keras", custom_objects=custom_objects)
    print("Evaluate on test data")
    
    for i in range(10):
        X, y_real = test_generator.__getitem__(i)
        
        
        print("y predicted shape:", y_real.shape)
        print("X shape:", X.shape)
        
        
        data = model1.predict(X)
        data_real=y_real
        
        z_slice = 64
        
        z_slice_data = data[0, :, z_slice, :, 3]
        
        z_slice_data_real = data_real[0, :, z_slice, :, 0]
        
        # Create a single figure with two subplots
        fig, (ax_predicted, ax_real) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot predicted data
        ax_predicted.imshow(z_slice_data)
        ax_predicted.set_title('Predicted')
        
        # Plot real data
        ax_real.imshow(z_slice_data_real)
        ax_real.set_title('Real')
        
        plt.subplots_adjust(wspace=-0.2)
        plt.savefig('../Plot/predict/ ' + str(i) +' '+ str(current_time) + ' .png')
        plt.show()

