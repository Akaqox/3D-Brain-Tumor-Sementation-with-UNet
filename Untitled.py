#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import nibabel as nib 
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler()


# In[20]:


TRAIN_DATASET_PATH = 'base_dir/train_ds/'


# In[22]:


test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_00000/BraTS2021_00000_flair.nii.gz').get_fdata()
print(test_image_flair.max())


# In[23]:


print(test_image_flair.shape)


# In[24]:


test_image_flair=scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)


# In[ ]:





# In[25]:


print(test_image_flair.shape)


# In[26]:


print(test_image_flair.max())


# In[28]:


test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_00000/BraTS2021_00000_t1.nii.gz').get_fdata()
test_image_t1=scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz').get_fdata()
test_image_t1ce=scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_00000/BraTS2021_00000_t2.nii.gz').get_fdata()
test_image_t2=scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS2021_00000/BraTS2021_00000_seg.nii.gz').get_fdata()
test_mask=test_mask.astype(np.uint8)


# In[29]:


print(np.unique(test_mask))



# In[30]:


test_mask[test_mask==4] = 3  #Reassign mask values 4 to 3
print(np.unique(test_mask)) 


# In[37]:


import random
n_slice=random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:,:,n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:,:,n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[:,:,n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[:,:,n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()


# In[38]:


print(test_image_flair.shape)
print(test_image_t1ce.shape)
print(test_image_t2.shape)


# In[39]:


combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)


# In[40]:


print(combined_x.shape)


# In[41]:


combined_x=combined_x[56:184, 56:184, 13:141] #Crop to 128x128x128x4





# In[42]:


#Do the same for mask
test_mask = test_mask[56:184, 56:184, 13:141]


# In[43]:


print(combined_x.shape)


# In[44]:


print(test_mask.shape)


# In[45]:


n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()


# In[46]:


test_mask = to_categorical(test_mask, num_classes=4)


# In[47]:


print(test_mask.shape)


# In[49]:


t2_list = sorted(glob.glob('base_dir/train_ds///*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob('base_dir/train_ds///*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('base_dir/train_ds///*/*flair.nii.gz'))
mask_list = sorted(glob.glob('base_dir/train_ds///*/*seg.nii.gz'))


# In[ ]:


for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
    
      
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    #print(np.unique(temp_mask))
    
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
    #cropping x, y, and z
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask= to_categorical(temp_mask, num_classes=4)
        if(not os.path.isdir("BraTask_TrainingData")):
            os.mkdir("BraTask_TrainingData")
            os.mkdir("BraTask_TrainingData/images")
            os.mkdir("BraTask_TrainingData/masks")
           
        np.save('BraTask_TrainingData/images/image_'+str(img)+'.npy', temp_combined_images)
        np.save('BraTask_TrainingData/masks/mask_'+str(img)+'.npy', temp_mask)
        
    else:
        print("I am useless")   
   
     


# In[3]:


if(not os.path.isdir("BraTask_Validation_30")):
    os.mkdir("BraTask_Validation_30")
    os.mkdir("BraTask_Validation_30/images")
    os.mkdir("BraTask_Validation_30/masks")

t2_list = sorted(glob.glob('BraTask_Validation_30//*/*t2.nii'))
t1ce_list = sorted(glob.glob('BraTask_Validation_30//*/*t1ce.nii'))
flair_list = sorted(glob.glob('BraTask_Validation_30//*/*flair.nii'))
mask_list = sorted(glob.glob('BraTask_Validation_30//*/*seg.nii'))


# In[8]:


for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
      
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    #print(np.unique(temp_mask))
    
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
    #cropping x, y, and z
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save('BraTask_ValidationData/images/image_'+str(img)+'.npy', temp_combined_images)
        np.save('BraTask_ValidationData/masks/mask_'+str(img)+'.npy', temp_mask)
        
    else:
        print("I am useless")   
   


# In[ ]:



