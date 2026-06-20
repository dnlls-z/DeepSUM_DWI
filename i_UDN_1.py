import tensorflow as tf
import numpy as np
import time
import glob
import scipy
import argparse
import sys
import os
import random
import json
import shutil
import warnings
import h5py
import progressbar

from PIL import Image

from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from dataclasses import dataclass

from scipy.ndimage import fourier_shift

from skimage.transform import rescale
from skimage.registration import phase_cross_correlation
from skimage import io
import skimage

from tensorflow.python.client import timeline

@dataclass(frozen=True)
class DataLoaderConfig:
    nickname: str
    #numband: int
    hiresl_h: int
    hiresl_w: int


def load_dataloader_config(config_path: str) -> DataLoaderConfig:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    return DataLoaderConfig(
        nickname=cfg["nickname"],
        #numband=cfg["numband"],
        hiresl_h=cfg["hiresl_h"],
        hiresl_w=cfg["hiresl_w"],
    )

#### UTILS #####
#### UTILS #####
#### UTILS #####

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def load_from_directory_to_pickle(base_dir,out_dir,band='NIR'):

    out_dir=out_dir.rstrip()
    
    train_dir = os.path.join(base_dir, 'train/'+band)
    dir_list=glob.glob(train_dir+'/imgset*')
    
    dir_list.sort()
    
    input_images_LR = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/LR*.png'))] 
                             for dir_name in dir_list ])
    
    input_images_LR.dump(out_dir+'/'+'LR_dataset_'+band+'.npy')
    
    input_images_HR = np.array([io.imread(glob.glob(dir_name+'/HR.png')[0]) for dir_name in dir_list ])
    
    input_images_HR.dump(out_dir+'/'+'HR_dataset_'+band+'.npy')
    
    mask_HR = np.array([io.imread(glob.glob(dir_name+'/SM.png')[0]) for dir_name in dir_list ],dtype=np.bool)
    
    mask_HR.dump(out_dir+'/'+'HR_mask_'+band+'.npy')
    
    mask_LR = np.array([[io.imread(fname)for fname in sorted(glob.glob(dir_name+'/QM*.png'))] 
                             for dir_name in dir_list ])
    
    imsetcount=0
    for imset in mask_LR:
        imcount=0
        for im in imset:
            mask_LR[imsetcount][imcount] = np.array(im, bool)
            imcount=imcount+1
        imsetcount=imsetcount+1
    
    mask_LR.dump(out_dir+'/'+'LR_mask_'+band+'.npy')
        
        
    train_dir = os.path.join(base_dir, 'test/'+band)
    dir_list=glob.glob(train_dir+'/imgset*')
    dir_list.sort()
    test_images_LR = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/LR*.png'))] 
                             for dir_name in dir_list ])
    
    test_images_LR.dump(out_dir+'/'+'LR_test_'+band+'.npy')
    
    test_mask_LR = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/QM*.png'))] 
                             for dir_name in dir_list ])                              
    
    imsetcount=0
    for imset in test_mask_LR:
        imcount=0
        for im in imset:
            test_mask_LR[imsetcount][imcount] = np.array(im, bool)
            imcount=imcount+1
        imsetcount=imsetcount+1
    
    test_mask_LR.dump(out_dir+'/'+'LR_mask_'+band+'_test.npy')
    

def registration_imageset_against_best_image_without_union_mask(batch_training,batch_training_mask, upsample_factor):
    batch_training_registered=[]
    batch_training_mask_registered=[]
    new_index_orders=[]
    
    shifts=[]
    
    for i in range(len(batch_training)):
        
        batch_training[i]=np.array(batch_training[i])
        
        imageset_training_registered=np.empty_like(batch_training[i])
        imageset_training_mask_registered=np.empty_like(batch_training_mask[i])
        imageset_shifts=np.empty([batch_training[i].shape[0],2])
        
        new_index_order=np.empty([batch_training[i].shape[0]],dtype='int16')

        index=np.argsort(np.sum(np.array(batch_training_mask[i]),axis=(1,2)))[::-1][0]
        z=0
        
        for j in range(batch_training[i].shape[0]):
            reference_image=batch_training[i][index]
            
            if j==index:
                j_index=0
                z=1
            else:
                j_index=j+1-z
                
            new_index_order[j_index]=j
            
            shifted_image=batch_training[i][j]
            
           
            shift, error, _ = phase_cross_correlation(reference_image.squeeze(), shifted_image.squeeze(), upsample_factor=upsample_factor, normalization=None)
            imageset_shifts[j_index]=np.asarray(shift)

            shifted_image_not_masked=batch_training[i][j]
            corrected_image = fourier_shift(np.fft.fftn(shifted_image_not_masked.squeeze()), imageset_shifts[j_index])
            corrected_image = np.fft.ifftn(corrected_image)
            imageset_training_registered[j_index]=corrected_image

            shifted_mask=batch_training_mask[i][j]
            corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), imageset_shifts[j_index])
            corrected_mask = np.fft.ifftn(corrected_mask)
            imageset_training_mask_registered[j_index]=corrected_mask
    
        imageset_training_mask_registered=np.round(imageset_training_mask_registered)
        imageset_training_mask_registered=imageset_training_mask_registered.astype('bool')

        batch_training_registered.append(imageset_training_registered)
        batch_training_mask_registered.append(imageset_training_mask_registered)
        shifts.append(imageset_shifts)
        new_index_orders.append(new_index_order)
            
    return batch_training_registered,batch_training_mask_registered,shifts,new_index_orders


def upsampling_mask(masks,scale=3):
    masks_images=np.empty([masks.shape[0],
                        masks.shape[1],
                        masks.shape[2]*scale,
                        masks.shape[3]*scale],dtype='bool')
    
    for i in range(masks.shape[0]):
        
        upsampled_image=np.zeros( (masks.shape[2]*scale,
                                   masks.shape[3]*scale), 
                                 dtype=np.bool)
        for j in range(masks.shape[1]):
            upsampled_image=rescale(masks[i,j].squeeze(), 
                                    scale=3, 
                                    order=0,
                                    mode='constant',
                                    anti_aliasing=False,
                                    preserve_range=True)
            upsampled_image=upsampled_image.astype('bool')
            masks_images[i,j]=upsampled_image
            
    return masks_images


def upsampling_mask_all_imageset(masks,scale=3):
    
    height=masks[0][0].shape[0]
    width=masks[0][0].shape[1]
    masks_images=np.empty([masks.shape[0]],dtype=object)
    
    for i in range(masks.shape[0]):
        
        list_maskset=[]
        
        upsampled_image=np.zeros( (height*scale,
                                   width*scale), 
                                 dtype='bool')
        for j in range(len(masks[i])):
            upsampled_image=rescale(masks[i][j].squeeze(), 
                                    scale=3, 
                                    order=0,
                                    mode='constant',
                                    anti_aliasing=False,
                                    preserve_range=True)
            
            upsampled_image=np.round(upsampled_image).astype('bool')
            list_maskset.append(upsampled_image)
            
        masks_images[i]=list_maskset
            
    return masks_images


def upsampling_without_aggregation_all_imageset(batch_training_to_up,scale=3):
    
    height=batch_training_to_up[0][0].shape[0]
    width=batch_training_to_up[0][0].shape[1]
    
    SR_images=np.empty([batch_training_to_up.shape[0]],dtype=object)
    
    for i in range(batch_training_to_up.shape[0]):
        
        list_imageset=[]
        
        upsampled_image=np.zeros( (height*scale,
                                   width*scale), 
                                 dtype=np.float32)
        
        for j in range(len(batch_training_to_up[i])):
            upsampled_image=rescale(batch_training_to_up[i][j].squeeze(), 
                                    scale=3, 
                                    order=3,
                                    mode='edge',
                                    anti_aliasing=False,
                                    preserve_range=True)
            upsampled_image=upsampled_image.astype('float32')
            list_imageset.append(upsampled_image)
            
        SR_images[i]=list_imageset
            
    return SR_images    


he_normal_init =tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,mode="fan_avg", distribution=("uniform" if False else "truncated_normal"), seed=1234)


def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]

    with tf.compat.v1.variable_scope(name):
        beta = tf.compat.v1.get_variable('beta', fdim, initializer=tf.compat.v1.constant_initializer(value=0.0))
        gamma = tf.compat.v1.get_variable('gamma', fdim, initializer=tf.compat.v1.constant_initializer(value=1.0))
        moving_mean = tf.compat.v1.get_variable('moving_mean', fdim, initializer=tf.compat.v1.constant_initializer(value=0.0), trainable=False)
        moving_variance = tf.compat.v1.get_variable('moving_variance', fdim, initializer=tf.compat.v1.constant_initializer(value=0.0), trainable=False)
  
        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 1e-3)


def InstanceNorm(input, axis=[2,3] , decay=0.999, name='InstanceNorm',trainable=True, reuse=tf.compat.v1.AUTO_REUSE):
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    fdim = input.get_shape()[-1:]

    with tf.compat.v1.variable_scope(name, reuse=reuse):
        beta = tf.compat.v1.get_variable('beta', fdim , dtype=tf.float32,initializer=tf.compat.v1.constant_initializer(value=0.0),trainable=trainable)
        gamma = tf.compat.v1.get_variable('gamma', fdim, dtype=tf.float32,initializer=tf.compat.v1.constant_initializer(value=1.0),trainable=trainable)
        
        instance_mean, instance_variance = tf.nn.moments(input, axis ,keepdims=True)
    
    return tf.nn.batch_normalization(input, instance_mean, instance_variance, beta, gamma, 1e-3)


def Conv3D(input, kernel_shape, strides, padding, scope_name='Conv3d', W_initializer=he_normal_init, trainable=True,bias=True):
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        W = tf.compat.v1.get_variable("W", kernel_shape, dtype=tf.float32,initializer=W_initializer,trainable=trainable)
        if bias is True:
            b = tf.compat.v1.get_variable("b", (kernel_shape[-1]),dtype=tf.float32,initializer=tf.compat.v1.constant_initializer(value=0.0),trainable=trainable)
        else:
            b = 0
        
    return tf.nn.conv3d(input, W, strides, padding) + b


def Conv2D(inputs, kernel_shape, strides, padding, scope_name='Conv2d',W_initializer=he_normal_init, bias=True,trainable=True):

    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        kernels=tf.compat.v1.get_variable('W',shape=kernel_shape,dtype=tf.float32,initializer=W_initializer,trainable=trainable)
        
        if bias is True:
            biases=tf.compat.v1.get_variable('b',shape=[kernel_shape[-1]],dtype=tf.float32,initializer=tf.compat.v1.constant_initializer(),trainable=trainable)
        else:
            biases = 0
        conv=tf.nn.bias_add(tf.nn.conv2d(inputs,filters=kernels,strides=strides,padding=padding),biases)   

    return conv


def Conv2D_transposed(inputs, kernel_shape, output_shape,strides, padding, scope_name='Conv2d',W_initializer=he_normal_init, bias=True):

    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        kernels=tf.compat.v1.get_variable('W',shape=kernel_shape,dtype=tf.float32,initializer=W_initializer)
        
        if bias is True:
            biases=tf.compat.v1.get_variable('b',shape=[kernel_shape[-2]],dtype=tf.float32,initializer=tf.compat.v1.constant_initializer())
        else:
            biases = 0
        conv=tf.nn.bias_add(tf.nn.conv2d_transpose(inputs,kernels,output_shape,strides=strides,padding=padding),biases)   

    return conv


def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]])
    
    y = tf.nn.depth_to_space(x, block_size)
    
    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


### DATALOADER ###
### DATALOADER ###
### DATALOADER ###


def new_coordinate(shape_original=[128,128],patch_size=[32,32]):
    image_size = shape_original
    portion_size = patch_size

    x1 = random.randint(0, image_size[0]-portion_size[0]-1)
    y1 = random.randint(0, image_size[1]-portion_size[1]-1)

    x2, y2 = x1+portion_size[0], y1+portion_size[1]
    
    return (x1,y1),(x2,y2)


def load_training_best(input_images_LR,mask_LR,top=9):
    indexes=[np.argsort(np.sum(np.array(image_set),axis=(1,2)))[::-1][0:top] for image_set in mask_LR]
    
    
    batch_training=np.array([np.array(l)[indexes[i]] for i,l in enumerate(input_images_LR)])
    sh=batch_training.shape
    batch_training=batch_training.reshape([-1,sh[1],sh[2],sh[3],1])
    
    
    batch_training_mask=np.array([np.array(l)[indexes[i]] for i,l in enumerate(mask_LR)])
    sh=batch_training_mask.shape
    batch_training_mask=batch_training_mask.reshape([-1,sh[1],sh[2],sh[3],1])

    return batch_training,batch_training_mask


def load_training_random(input_images_LR,mask_LR,num_images=9):
    indexes=[random.sample(list(range(0,len(image_set))),num_images) for image_set in mask_LR]
    
    
    batch_training=np.array([np.array(l)[indexes[i]] for i,l in enumerate(input_images_LR)])
    sh=batch_training.shape
    batch_training=batch_training.reshape([-1,sh[1],sh[2],sh[3],1])
    
    
    batch_training_mask=np.array([np.array(l)[indexes[i]] for i,l in enumerate(mask_LR)])
    sh=batch_training_mask.shape
    batch_training_mask=batch_training_mask.reshape([-1,sh[1],sh[2],sh[3],1])

    return batch_training,batch_training_mask


def load_training_first(input_images_LR,mask_LR,top=9):
    batch_training=np.array([np.array(l)[0:top] for i,l in enumerate(input_images_LR)])
    sh=batch_training.shape
    batch_training=batch_training.reshape([-1,sh[1],sh[2],sh[3],1])
    
    
    batch_training_mask=np.array([np.array(l)[0:top] for i,l in enumerate(mask_LR)])
    sh=batch_training_mask.shape
    batch_training_mask=batch_training_mask.reshape([-1,sh[1],sh[2],sh[3],1])

    return batch_training,batch_training_mask
    

def create_patch_dataset_return_shifts(input_images_upsample,input_images_HR,mask_upsample,mask_HR,shifts,patch_size=96,num_patches_per_set=100,scale=3,smart_patching=False):
    
    input_images_upsample_patch=[]
    input_images_HR_patch=[]
    mask_upsample_patch=[]
    mask_HR_patch=[]
    coordinates=[]
    shifts_patch=[]
    
    tot_num_patches=num_patches_per_set
    max_trial=100000
    scale=scale
    n_samples=len(input_images_upsample)
    
    shape_original=[input_images_upsample[0][0].shape[0],input_images_upsample[0][0].shape[1]]
    bar = progressbar.ProgressBar(maxval=n_samples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    
    i=0
    for j in range(n_samples):
        
        image_set=input_images_upsample[j]
        mask_set=mask_upsample[j]
        upsample_images_set=np.array(image_set)
        upsample_mask_set=np.array(mask_set)

        image_HR=input_images_HR[j]
        mask_image_HR=mask_HR[j]
        
        current_num_patches=0
        num_trial=0
        
        
        coordinates_for_one_set=[]
        
        
        while True:
            if current_num_patches>=tot_num_patches or num_trial>=max_trial:
                break
            

            
            x,y=new_coordinate(shape_original=shape_original,patch_size=[patch_size,patch_size])
            num_trial+=1
            patches_upsample=upsample_images_set[:,x[0]:y[0],x[1]:y[1]]
            patches_HR=image_HR[x[0]*scale:y[0]*scale,x[1]*scale:y[1]*scale]
            

            patch_masks_upsample=upsample_mask_set[:,x[0]:y[0],x[1]:y[1]]
            
            checked_upsample=[(np.sum(patch_mask)/(patch_size**2))>0.70 for patch_mask in patch_masks_upsample]
            
            patch_masks_HR=mask_image_HR[x[0]*scale:y[0]*scale,x[1]*scale:y[1]*scale]
            
            checked_HR=((np.sum(patch_masks_HR)/((patch_size*scale)**2))>0.85 )
            
            if smart_patching:
                checked_LR=(sum(checked_upsample)>=9)
            else:
                checked_LR=all(checked_upsample)
            
            if  checked_LR and checked_HR:
                input_images_upsample_patch.append(patches_upsample)
                input_images_HR_patch.append(patches_HR)
                mask_upsample_patch.append(patch_masks_upsample)
                mask_HR_patch.append(patch_masks_HR)
                coordinates_for_one_set.append((x,y))
                shifts_patch.append(np.copy(shifts[j]))
                current_num_patches+=1
        
        coordinates.append(coordinates_for_one_set)
        
        
        bar.update(j+1)
    
    bar.finish() 
    dataset_patch=defaultdict()
    dataset_patch['training_patch']=np.array(input_images_upsample_patch)
    dataset_patch['training_mask_patch']=np.array(mask_upsample_patch)
    dataset_patch['training_y_patch']=np.array(input_images_HR_patch)
    dataset_patch['training_mask_y_patch']=np.array(mask_HR_patch)
    dataset_patch['shifts']=shifts_patch
    dataset_patch['coordinates']=coordinates
    
    return  dataset_patch


def load_dataset(dl_config,path,n_chuncks,band='NIR',num_images=9,how='best'):
    nickname = dl_config.nickname

    input_images_LR_valid=np.load(path+'dataset_{0}_{1}_s_LR_valid.npy'.format(band, nickname),allow_pickle=True)
    input_images_HR_valid=np.load(path+'dataset_{0}_{1}_s_HR_valid.npy'.format(band, nickname),allow_pickle=True)
    mask_LR_valid=np.load(path+'dataset_{0}_mask_{1}_s_LR_valid.npy'.format(band, nickname),allow_pickle=True)
    mask_HR_valid=np.load(path+'dataset_{0}_mask_{1}_s_HR_valid.npy'.format(band, nickname),allow_pickle=True)
    
    shifts_valid=np.load(path+'shifts_valid_{0}_s_{1}.npy'.format(nickname, band),allow_pickle=True)

    
    if how=='first':
        batch_validation,batch_validation_mask=load_training_first(input_images_LR_valid,mask_LR_valid,top=9)
    elif how=='best':
       
        indexes=[np.argsort(np.sum(np.array(image_set[1:]),axis=(1,2)))[::-1][0:8]+1 for image_set in mask_LR_valid]
        indexes=[np.append(0,indexes_imageset) for indexes_imageset in indexes]
                
        batch_validation=np.array([image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_valid,indexes)])
        batch_validation=np.expand_dims(batch_validation,axis=-1)
        
        batch_validation_mask=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_valid,indexes)])
        batch_validation_mask=np.expand_dims(batch_validation_mask,axis=-1)
        
        shifts_valid=np.array([shifts_set[indexes_set] for shifts_set,indexes_set in zip(shifts_valid,indexes)])
        shifts_valid=-shifts_valid
        
    
    sh=input_images_HR_valid.shape
    batch_validation_y=input_images_HR_valid.reshape([-1,1,sh[1],sh[2],1])
    
    sh=mask_HR_valid.shape
    batch_mask_y_valid=mask_HR_valid.reshape([-1,1,sh[1],sh[2],1])
    

    norm_validation = np.load(
        f"{path}norm_{nickname}_s_{band}.npy",
        allow_pickle=True
    )

    dataset=defaultdict()

    dataset['validation']=batch_validation
    dataset['validation_mask']=batch_validation_mask
    dataset['validation_y']=batch_validation_y
    dataset['validation_mask_y']=batch_mask_y_valid
    
    dataset['shifts_valid']=shifts_valid
    dataset['norm_validation']=norm_validation
    
    
    pickle_indexes=np.array([i for i in range(0,n_chuncks)])
    np.random.shuffle(pickle_indexes)
    
    for i in pickle_indexes:
    
        input_images_LR_patch=np.load(path+'{0}_dataset_{1}_patch_{2}_s_LR.npy'.format(i,band,nickname),allow_pickle=True)
        input_images_HR_patch=np.load(path+'{0}_dataset_{1}_patch_{2}_s_HR.npy'.format(i,band,nickname),allow_pickle=True)
        mask_LR_patch=np.load(path+'{0}_dataset_{1}_patch_mask_{2}_s_LR.npy'.format(i,band,nickname),allow_pickle=True)
        mask_HR_patch=np.load(path+'{0}_dataset_{1}_patch_mask_{2}_s_HR.npy'.format(i,band,nickname),allow_pickle=True)
        shifts=np.load(path+'{0}_shifts_patch_{1}_s_{2}.npy'.format(i,nickname,band),allow_pickle=True)


        if how=='first':
            batch_training,batch_training_mask=load_training_first(input_images_LR_patch,mask_LR_patch,top=num_images)
        elif how=='best':
            indexes=[np.argsort(np.sum(np.array(image_set[1:]),axis=(1,2)))[::-1][0:8]+1 for image_set in mask_LR_patch]
            indexes=[np.append(0,indexes_imageset) for indexes_imageset in indexes]
            
            batch_training=np.array([image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_patch,indexes)])
            batch_training=np.expand_dims(batch_training,axis=-1)
            batch_training_mask=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_patch,indexes)])
            batch_training_mask=np.expand_dims(batch_training_mask,axis=-1)
            shifts=np.array([shifts_set[indexes_set] for shifts_set,indexes_set in zip(shifts,indexes)])
            shifts=-shifts
        
        
        sh=input_images_HR_patch.shape
        batch_training_y=input_images_HR_patch.reshape([-1,1,sh[1],sh[2],1])
        sh=mask_HR_patch.shape
        batch_mask_y_train=mask_HR_patch.reshape([-1,1,sh[1],sh[2],1])
        
        dataset['training']=batch_training
        dataset['training_mask']=batch_training_mask
        dataset['training_y']=batch_training_y
        dataset['training_mask_y']=batch_mask_y_train
        dataset['shifts']=shifts
        

        yield dataset
    

def load_dataset_best9(dl_config,path,n_chuncks,band='NIR'):
    
    
    nickname = dl_config.nickname

    
    batch_validation=np.load(path+'dataset_{0}_LR_{1}_valid_best9.npy'.format(band,nickname),allow_pickle=True)
    batch_validation_y=np.load(path+'dataset_{0}_HR_{1}_valid_best9.npy'.format(band,nickname),allow_pickle=True)
    batch_validation_mask=np.load(path+'dataset_{0}_mask_{1}_LR_valid_best9.npy'.format(band,nickname),allow_pickle=True)
    batch_mask_y_valid=np.load(path+'dataset_{0}_mask_{1}_HR_valid_best9.npy'.format(band,nickname),allow_pickle=True)
    
    shifts_valid=np.load(path+'shifts_{0}_valid_{1}_best9.npy'.format(nickname,band),allow_pickle=True)
    


    norm_validation = np.load(
        f"{path}norm_{nickname}_{band}.npy",
        allow_pickle=True
    )
   

    dataset=defaultdict()

    dataset['validation']=batch_validation
    dataset['validation_mask']=batch_validation_mask
    dataset['validation_y']=batch_validation_y
    dataset['validation_mask_y']=batch_mask_y_valid
    
    dataset['shifts_valid']=shifts_valid
    dataset['norm_validation']=norm_validation
    
    
    pickle_indexes=np.array([i for i in range(0,n_chuncks)])
    np.random.shuffle(pickle_indexes)
    
    for i in pickle_indexes:
    
        batch_training=np.load(path+'{0}_dataset_{1}_patch_{2}_LR_best9.npy'.format(i,band,nickname),allow_pickle=True)
        batch_training_y=np.load(path+'{0}_dataset_{1}_patch_{2}_HR_best9.npy'.format(i,band,nickname),allow_pickle=True)
        batch_training_mask=np.load(path+'{0}_dataset_{1}_patch_mask_{2}_LR_best9.npy'.format(i,band,nickname),allow_pickle=True)
        batch_mask_y_train=np.load(path+'{0}_dataset_{1}_patch_mask_{2}_HR_best9.npy'.format(i,band,nickname),allow_pickle=True)
        shifts=np.load(path+'{0}_shifts_{1}_patch_{2}_best9.npy'.format(i,nickname,band),allow_pickle=True)
        
        dataset['training']=batch_training
        dataset['training_mask']=batch_training_mask
        dataset['training_y']=batch_training_y
        dataset['training_mask_y']=batch_mask_y_train
        dataset['shifts']=shifts
        

        yield dataset


def load_testset_preprocesses(dl_config,path,band='NIR',how='best',num_images=9):
    
    nickname = dl_config.nickname

    input_images_LR_test=np.load(path+'dataset_{0}_{1}_s_LR_test.npy'.format(band,nickname),allow_pickle=True)
    mask_LR_test=np.load(path+'dataset_{0}_{1}_s_mask_LR_test.npy'.format(band,nickname),allow_pickle=True)
    shifts_test=np.load(path+'shifts_test_{0}_s_{1}.npy'.format(nickname,band),allow_pickle=True)
    
    if how=='first':
        batch_test,batch_test_mask=load_training_first(input_images_LR_test,mask_LR_test,top=num_images)
    elif how=='best':
        indexes=[np.argsort(np.sum(np.array(image_set[1:]),axis=(1,2)))[::-1][0:8]+1 for image_set in mask_LR_test]
        indexes=[np.append(0,indexes_imageset) for indexes_imageset in indexes]
        
        batch_test=np.array([image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_test,indexes)])
        batch_test=np.expand_dims(batch_test,axis=-1)
        batch_test_mask=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_test,indexes)])
        batch_test_mask=np.expand_dims(batch_test_mask,axis=-1)
        shifts_test=np.array([shifts_set[indexes_set] for shifts_set,indexes_set in zip(shifts_test,indexes)])
        shifts_test=-shifts_test
    
        
    dataset=defaultdict()
    dataset['test']=batch_test
    dataset['test_mask']=batch_test_mask
    dataset['shifts_test']=shifts_test
    
    return dataset



##### DEEPSUM_NET #####
##### DEEPSUM_NET #####
##### DEEPSUM_NET #####


class SR_network(object):
    def __init__(self, config):

        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.tensorboard_dir = config['tensorboard_dir']

        self.skip_step = config['skip_step']

        self.channels = config['channels']
        self.T_in = config['T_in']
        self.R = config['R']
        self.full = config['full']

        self.patch_size_HR = config['patch_size_HR']
        self.patch_size_LR = config['patch_size_LR']
        self.border = config['border']
        self.spectral_band = config['spectral_band']

        self.dyn_filter_size = 9

        self.RegNet_pretrain_dir = config['RegNet_pretrain_dir']
        self.SISRNet_pretrain_dir = config['SISRNet_pretrain_dir']
        self.dataset_path = config['dataset_path']
        self.n_chunks = config['n_chunks']

        self.nickname = config['nickname']
        #self.numband = config['numband']
        self.hiresl_h = config['hiresl_h']
        self.hiresl_w = config['hiresl_w']

        self.data_cfg = DataLoaderConfig(
            nickname=self.nickname,
            #numband=self.numband,
            hiresl_h=self.hiresl_h,
            hiresl_w=self.hiresl_w,
        )

        self.TRAIN_REGIST_NET = True
        self.TRAIN_UPSAMPLING_NET = True
        self.TRAIN_FUSION_NET = True

        self.placeholder()

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options))

        self.mu = config['mu']
        self.sigma = config['sigma']
        self.sigma_rescaled = config['sigma_rescaled']


    def placeholder(self):

        self.x = tf.compat.v1.placeholder('float32', shape=[None, None, None, None, None], name='x')

        self.y = tf.compat.v1.placeholder('float32', shape=[None, 1, None, None, 1], name='y')
        self.mask_y = tf.compat.v1.placeholder('float32', shape=[None, 1, None, None, 1], name='mask_y')
        self.y_filters = tf.compat.v1.placeholder('float32', shape=[None, None, self.dyn_filter_size ** 2],
                                                  name='y_filters')
        self.fill_coeff = tf.compat.v1.placeholder(tf.float32, shape=[None, self.T_in, self.T_in, None, None, 1],
                                                   name='fill_coeff')
        self.norm_baseline = tf.compat.v1.placeholder('float32', shape=[None, 1], name='norm_baseline')


    def get_data(self):
        with tf.compat.v1.name_scope('data'):

            try:
                print('Retrieving portion of training...')
                dataset_dict = next(self.gen)
            except StopIteration:
                return 0

            self.batch_training = dataset_dict['training']
            print(self.batch_training.shape)
            self.batch_training_mask = dataset_dict['training_mask']
            self.batch_training_y = dataset_dict['training_y']
            self.batch_mask_train_y = dataset_dict['training_mask_y']
            self.shifts = dataset_dict['shifts']

            self.batch_validation = dataset_dict['validation']
            self.batch_validation_mask = dataset_dict['validation_mask']
            self.batch_validation_y = dataset_dict['validation_y']
            self.batch_mask_valid_y = dataset_dict['validation_mask_y']

            self.shifts_valid = dataset_dict['shifts_valid']
            self.norm_validation = dataset_dict['norm_validation']

            self.batch_training_mask = np.round(self.batch_training_mask)
            self.batch_validation_mask = np.round(self.batch_validation_mask)
            self.batch_training_mask = self.batch_training_mask.astype('bool')
            self.batch_validation_mask = self.batch_validation_mask.astype('bool')

            for i in range(np.shape(self.batch_training_mask)[0]):
                shifted_mask_imageset = np.zeros_like(self.batch_training_mask[i], dtype='bool')
                for j in range(self.batch_training_mask[i].shape[0]):
                    shifted_mask = self.batch_training_mask[i][j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), -self.shifts[i][j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape(
                        [1, np.shape(self.batch_training_mask)[2], np.shape(self.batch_training_mask)[3], 1])
                    shifted_mask_imageset[j] = np.round(corrected_mask)

                self.batch_training_mask[i] = shifted_mask_imageset

            for i in range(np.shape(self.batch_validation_mask)[0]):
                shifted_mask_imageset = np.zeros_like(self.batch_validation_mask[i], dtype='bool')
                for j in range(self.batch_validation_mask[i].shape[0]):
                    shifted_mask = self.batch_validation_mask[i][j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), -self.shifts_valid[i][j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape(
                        [1, np.shape(self.batch_validation_mask)[2], np.shape(self.batch_validation_mask)[3], 1])
                    shifted_mask_imageset[j] = np.round(corrected_mask)

                self.batch_validation_mask[i] = shifted_mask_imageset

            sh = self.batch_training_mask.shape
            self.fill_coeff_train = np.ones([sh[0], sh[1], sh[1], sh[2], sh[3], sh[4]], dtype='bool')
            for i in range(0, 9):
                self.fill_coeff_train[:, :, i] = np.expand_dims(self.batch_training_mask[:, i], axis=1)

            for i in range(0, 9):
                for j in range(i + 1, 9):
                    rows_indexes = [k for k in range(0, 9) if k != (j)]
                    self.fill_coeff_train[:, rows_indexes, j] = self.fill_coeff_train[
                                                                    :, rows_indexes, j] * np.expand_dims(
                        1 - self.batch_training_mask[:, i], axis=1)

            for i in range(1, 9):
                self.fill_coeff_train[:, i, 0:i] = self.fill_coeff_train[:, i, 0:i] * np.expand_dims(
                    1 - self.batch_training_mask[:, i], axis=1)

            f = np.sum(self.fill_coeff_train, axis=2)
            self.fill_coeff_train[:, range(9), range(9), :, :, :] = self.fill_coeff_train[
                                                                        :, range(9), range(9), :, :, :] + \
                                                                    np.logical_not(f)[:, range(9), :, :, :]

            sh = self.batch_validation_mask.shape
            self.fill_coeff_valid = np.ones([sh[0], sh[1], sh[1], sh[2], sh[3], sh[4]], dtype='bool')
            for i in range(0, 9):
                self.fill_coeff_valid[:, :, i] = np.expand_dims(self.batch_validation_mask[:, i], axis=1)

            for i in range(0, 9):
                for j in range(i + 1, 9):
                    rows_indexes = [k for k in range(0, 9) if k != (j)]
                    self.fill_coeff_valid[:, rows_indexes, j] = self.fill_coeff_valid[
                                                                    :, rows_indexes, j] * np.expand_dims(
                        1 - self.batch_validation_mask[:, i], axis=1)

            for i in range(1, 9):
                self.fill_coeff_valid[:, i, 0:i] = self.fill_coeff_valid[:, i, 0:i] * np.expand_dims(
                    1 - self.batch_validation_mask[:, i], axis=1)

            f = np.sum(self.fill_coeff_valid, axis=2)
            self.fill_coeff_valid[:, range(9), range(9), :, :, :] = self.fill_coeff_valid[
                                                                        :, range(9), range(9), :, :, :] + \
                                                                    np.logical_not(f)[:, range(9), :, :, :]

            self.batch_training = (self.batch_training - self.mu) / self.sigma
            self.batch_training_y = (self.batch_training_y - self.mu) / self.sigma_rescaled
            self.batch_validation = (self.batch_validation - self.mu) / self.sigma
            self.batch_validation_y = (self.batch_validation_y - self.mu) / self.sigma_rescaled

            self.batch_training_norm = self.batch_training
            self.batch_validation_norm = self.batch_validation
            self.batch_training_y_norm = self.batch_training_y
            self.batch_validation_y_norm = self.batch_validation_y

            self.y_filters_dyn = np.zeros([self.shifts.shape[0], np.shape(self.batch_training_norm)[1],
                                           self.dyn_filter_size,
                                           self.dyn_filter_size])

            self.shifts = np.array([shift[0:self.T_in] for shift in self.shifts])
            self.shifts = self.shifts + int(self.dyn_filter_size / 2)
            self.shifts = self.shifts.astype('int32')

            for i, shift in enumerate(self.shifts):
                self.y_filters_dyn[
                    i, list(range(0, np.shape(self.batch_training_norm)[1])), self.shifts[i, :, 0], self.shifts[
                        i, :, 1]] = 1

            self.y_filters_dyn = np.reshape(self.y_filters_dyn,
                                            [-1,
                                             np.shape(self.batch_training_norm)[1],
                                             self.dyn_filter_size ** 2])

            self.y_filters_dyn = self.y_filters_dyn[:, 1:, :]

            self.y_filters_valid_dyn = np.zeros([self.shifts_valid.shape[0], np.shape(self.batch_validation_norm)[1],
                                                 self.dyn_filter_size,
                                                 self.dyn_filter_size])

            self.shifts_valid = np.array([shift[0:self.T_in] for shift in self.shifts_valid])
            self.shifts_valid = self.shifts_valid + int(self.dyn_filter_size / 2)
            self.shifts_valid = self.shifts_valid.astype('int32')

            for i, shift in enumerate(self.shifts_valid):
                self.y_filters_valid_dyn[
                    i, list(range(0, np.shape(self.batch_validation_norm)[1])), self.shifts_valid[i, :, 0],
                    self.shifts_valid[i, :, 1]] = 1

            self.y_filters_valid_dyn = np.reshape(self.y_filters_valid_dyn,
                                                  [-1,
                                                   np.shape(self.batch_validation_norm)[1],
                                                   self.dyn_filter_size ** 2])
            self.y_filters_valid_dyn = self.y_filters_valid_dyn[:, 1:, :]

    def loss(self):

        with tf.compat.v1.name_scope('loss') as scope:

            self.y_hat = self.logits

            s1 = tf.shape(self.y)
            s2 = tf.shape(self.y_hat)
            labels = tf.reshape(self.y, shape=[s1[0], s1[2], s1[3], s1[4]])
            predictions = tf.reshape(self.y_hat, shape=[s2[0], s2[2], s2[3], s2[4]])

            size_image = tf.shape(predictions)[1]
            cropped_predictions = predictions[
                :, self.border:size_image - self.border, self.border:size_image - self.border]

            X = []
            for i in range((2 * self.border) + 1):
                for j in range((2 * self.border) + 1):
                    cropped_labels = labels[
                        :, i:i + (size_image - (2 * self.border)), j:j + (size_image - (2 * self.border))]
                    cropped_mask_y = self.mask_y[
                        :, :, i:i + (size_image - (2 * self.border)), j:j + (size_image - (2 * self.border))]

                    cropped_predictions_masked = cropped_predictions * tf.squeeze(cropped_mask_y, axis=1)
                    cropped_labels_masked = cropped_labels * tf.squeeze(cropped_mask_y, axis=1)

                    b = (1.0 / tf.reduce_sum(cropped_mask_y, axis=[2, 3, 4])) * tf.reduce_sum(
                        cropped_labels_masked - cropped_predictions_masked, axis=[1, 2])
                    b = tf.reshape(b, [s1[0], 1, 1, 1])
                    corrected_cropped_predictions = cropped_predictions_masked + b
                    corrected_cropped_predictions = corrected_cropped_predictions * tf.squeeze(cropped_mask_y, axis=1)
                    corrected_mse = (1.0 / tf.reduce_sum(cropped_mask_y, axis=[2, 3, 4])) * tf.reduce_sum(
                        tf.square(cropped_labels_masked - corrected_cropped_predictions), axis=[1, 2])

                    X.append(corrected_mse)

            X = tf.stack(X)
            minim = tf.reduce_min(X, axis=0)
            mse = tf.reduce_mean(minim)

            self.loss = mse

    def optimize(self):

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        gradients, gradient_tensors = zip(*self.grads_and_vars)
        self.opt = optimizer.apply_gradients(self.grads_and_vars, global_step=self.gstep)

    def inference_FR(self):

        stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
        sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

        F = 64

        with tf.compat.v1.variable_scope('conv_up_0', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(self.x, sp, mode='REFLECT'), [1, 3, 3, self.channels, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_0', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_0', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_1', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_1', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_1', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_2', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_2', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_2', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_3', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_3', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_3', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_4', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_4', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_4', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_5', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_5', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_5', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_6', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_6', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_6', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_up_7', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1, 3, 3, F, F], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_up_7', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_up_7', trainable=self.TRAIN_UPSAMPLING_NET)
            x1 = tf.nn.leaky_relu(x1)

        self.x1_from_upsampling = tf.identity(x1)
        self.x1_to_filter = tf.identity(x1[:, 1:, :, :, :])

        self.references = tf.tile(x1[:, 0:1, :, :, :], [1, self.T_in, 1, 1, 1])

        x1 = tf.stack([x1, self.references], axis=2)
        self.f1 = x1
        sh = tf.shape(x1)
        x1 = tf.reshape(x1, [sh[0], sh[1] * sh[2], sh[3], sh[4], sh[5]])

        x1 = x1[:, 2:, :, :, :]

        F1 = 64
        F2 = 64

        with tf.compat.v1.variable_scope('Rconv1b', reuse=tf.compat.v1.AUTO_REUSE):
            t = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [2, 3, 3, F2, 128], [1, 2, 1, 1, 1], 'VALID',
                       scope_name='Rconv1b', trainable=self.TRAIN_REGIST_NET)
            t = tf.nn.leaky_relu(t)

        with tf.compat.v1.variable_scope('Rconv2b', reuse=tf.compat.v1.AUTO_REUSE):
            t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1, 3, 3, 128, F1], [1, 1, 1, 1, 1], 'VALID',
                       scope_name='Rconv2b', trainable=self.TRAIN_REGIST_NET)
            t = tf.nn.leaky_relu(t)

        with tf.compat.v1.variable_scope('Rconv3b', reuse=tf.compat.v1.AUTO_REUSE):
            t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1, 3, 3, F1, F1], [1, 1, 1, 1, 1], 'VALID', scope_name='Rconv3b',
                       trainable=self.TRAIN_REGIST_NET)
            t = tf.nn.leaky_relu(t)

        with tf.compat.v1.variable_scope('Rconv4b', reuse=tf.compat.v1.AUTO_REUSE):
            t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1, 3, 3, F1, F1], [1, 1, 1, 1, 1], 'VALID', scope_name='Rconv4b',
                       trainable=self.TRAIN_REGIST_NET)
            t = tf.nn.leaky_relu(t)

        with tf.compat.v1.variable_scope('Rconv7b', reuse=tf.compat.v1.AUTO_REUSE):
            t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1, 3, 3, F1, self.dyn_filter_size ** 2], [1, 1, 1, 1, 1],
                       'VALID', scope_name='Rconv7_b', trainable=self.TRAIN_REGIST_NET)

        t = tf.reduce_mean(t, axis=[2, 3])
        self.logits_filters = tf.identity(t)

        self.filters = tf.nn.softmax(t, axis=2)
        sh = tf.shape(self.filters)
        self.filters = tf.reshape(self.filters, [sh[0], sh[1], self.dyn_filter_size, self.dyn_filter_size])

        self.x_to_fuse = self.registration_dyn_filters(self.x1_to_filter, self.filters)
        self.x_registered = self.registration_dyn_filters(self.x[:, 1:, :, :, :], self.filters)

        self.x_to_fuse = tf.concat([self.x1_from_upsampling[:, 0:1, :, :, :], self.x_to_fuse], axis=1)
        self.x_registered = tf.concat([self.x[:, 0:1, :, :, :], self.x_registered], axis=1)

        self.fill_coeff_feature = tf.tile(self.fill_coeff, [1, 1, 1, 1, 1, F])
        self.x_to_fuse = tf.reduce_sum(
            self.fill_coeff_feature[:, :, :, :, :, :] * tf.expand_dims(self.x_to_fuse[:, :, :, :, :], axis=1), axis=2)
        self.x_registered = tf.reduce_sum(
            self.fill_coeff[:, :, :, :, :, :] * tf.expand_dims(self.x_registered[:, :, :, :, :], axis=1), axis=2)

        with tf.compat.v1.variable_scope('conv_fuse_0', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(self.x_to_fuse, sp, mode='REFLECT'), [3, 3, 3, F, F1], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_fuse_0', trainable=self.TRAIN_FUSION_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_fuse_0', trainable=self.TRAIN_FUSION_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_fuse_1', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [3, 3, 3, F1, F1], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_fuse_1', trainable=self.TRAIN_FUSION_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_fuse_1', trainable=self.TRAIN_FUSION_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_fuse_2', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [3, 3, 3, F1, F1], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_fuse_2', trainable=self.TRAIN_FUSION_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_fuse_2', trainable=self.TRAIN_FUSION_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_fuse_3', reuse=tf.compat.v1.AUTO_REUSE):
            x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [3, 3, 3, F1, F1], [1, 1, 1, 1, 1], 'VALID',
                        scope_name='conv_fuse_3', trainable=self.TRAIN_FUSION_NET)
            x1 = InstanceNorm(x1, axis=[2, 3], name='inst_norm_fuse_3', trainable=self.TRAIN_FUSION_NET)
            x1 = tf.nn.leaky_relu(x1)

        with tf.compat.v1.variable_scope('conv_fuse_4', reuse=tf.compat.v1.AUTO_REUSE):
            self.x1 = Conv3D(x1, [1, 1, 1, F1, 1], [1, 1, 1, 1, 1], 'VALID', scope_name='conv_fuse_4',
                             trainable=self.TRAIN_FUSION_NET)

        self.x_r = tf.reduce_mean(self.x_registered, axis=1)
        self.x_r = tf.expand_dims(self.x_r, axis=1)

        self.x_r = (self.x_r * self.sigma) / self.sigma_rescaled

        self.x1 += self.x_r

        self.SR_temp = self.x1
        self.logits = self.x1

    def registration_dyn_filters(self, x1_to_filter, filters):

        x1_to_filter = tf.identity(x1_to_filter)
        filters = tf.identity(filters)

        features_channels = tf.shape(x1_to_filter)[-1]
        depth = tf.shape(x1_to_filter)[1]
        batch = tf.shape(x1_to_filter)[0]
        batch_depth_dim = batch * depth

        sh = tf.shape(filters)
        filters = tf.reshape(filters, [sh[0] * sh[1], sh[2], sh[3]])
        filters = tf.expand_dims(filters, axis=3)
        filters = tf.transpose(filters, perm=[1, 2, 0, 3])
        filters = tf.tile(filters, [1, 1, 1, features_channels])
        sh = tf.shape(filters)
        filters = tf.reshape(filters, [sh[0], sh[1], sh[2] * sh[3]])
        filters = tf.expand_dims(filters, axis=-1)

        sh = tf.shape(x1_to_filter)
        x1_to_filter = tf.reshape(x1_to_filter, [sh[0] * sh[1], sh[2], sh[3], sh[4]])

        x1_to_filter = tf.transpose(x1_to_filter, perm=[1, 2, 0, 3])
        sh = tf.shape(x1_to_filter)
        x1_to_filter = tf.reshape(x1_to_filter, [sh[0], sh[1], sh[2] * sh[3]])
        x1_to_filter = tf.expand_dims(x1_to_filter, axis=0)

        pad_size = int((self.dyn_filter_size - 1) / 2)
        padding_reg = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
        x_to_fuse = tf.nn.depthwise_conv2d(tf.pad(x1_to_filter, padding_reg, mode='REFLECT'), filters, [1, 1, 1, 1],
                                           "VALID")
        sh = tf.shape(x_to_fuse)
        x_to_fuse = tf.reshape(x_to_fuse, [sh[0], sh[1], sh[2], batch_depth_dim, features_channels])
        sh = tf.shape(x_to_fuse)
        x_to_fuse = tf.reshape(x_to_fuse, [sh[0], sh[1], sh[2], batch, depth, sh[4]])
        x_to_fuse = tf.squeeze(x_to_fuse, axis=0)
        x_to_fuse = tf.transpose(x_to_fuse, perm=[2, 3, 0, 1, 4])

        return x_to_fuse

    def PSNR(self, norm=True):

        self.y_hat = self.logits

        y_hat = ((self.y_hat) * self.sigma_rescaled) + self.mu
        y = ((self.y) * self.sigma_rescaled) + self.mu

        s1 = tf.shape(y)
        s2 = tf.shape(y_hat)
        labels = tf.reshape(y, shape=[s1[0], s1[2], s1[3], s1[4]])
        predictions = tf.reshape(y_hat, shape=[s2[0], s2[2], s2[3], 1])

        size_image = tf.shape(predictions)[1]
        cropped_predictions = predictions[:, self.border:size_image - self.border, self.border:size_image - self.border]

        X = []
        for i in range((2 * self.border) + 1):
            for j in range((2 * self.border) + 1):
                cropped_labels = labels[
                    :, i:i + (size_image - (2 * self.border)), j:j + (size_image - (2 * self.border))]
                cropped_mask_y = self.mask_y[
                    :, :, i:i + (size_image - (2 * self.border)), j:j + (size_image - (2 * self.border))]

                cropped_predictions_masked = cropped_predictions * tf.squeeze(cropped_mask_y, axis=1)
                cropped_labels_masked = cropped_labels * tf.squeeze(cropped_mask_y, axis=1)

                b = (1.0 / tf.reduce_sum(cropped_mask_y, axis=[2, 3, 4])) * tf.reduce_sum(
                    cropped_labels_masked - cropped_predictions_masked, axis=[1, 2])
                b = tf.reshape(b, [s1[0], 1, 1, 1])
                corrected_cropped_predictions = cropped_predictions_masked + b
                corrected_cropped_predictions = corrected_cropped_predictions * tf.squeeze(cropped_mask_y, axis=1)
                corrected_mse = (1.0 / tf.reduce_sum(cropped_mask_y, axis=[2, 3, 4])) * tf.reduce_sum(
                    tf.square(cropped_labels_masked - corrected_cropped_predictions), axis=[1, 2])

                cPSNR = 10 * tf.math.log((65535 ** 2) / corrected_mse) / tf.math.log(10.0)
                X.append(cPSNR)

        X = tf.stack(X)

        if norm:
            max_cPSNR = tf.reduce_max(X, axis=0)
            max_cPSNR = tf.expand_dims(max_cPSNR, axis=-1)
            score = self.norm_baseline / max_cPSNR
            score = tf.reduce_mean(tf.cast(score, tf.float32))
            return score
        else:
            max_cPSNR = tf.reduce_max(X, axis=0)
            psnr = tf.reduce_mean(tf.cast(max_cPSNR, tf.float32))
            return psnr

    def SSIM(self):
        self.y_hat = self.logits

        y_hat = ((self.y_hat) * self.sigma_rescaled) + self.mu
        y = ((self.y) * self.sigma_rescaled) + self.mu

        s1 = tf.shape(y)
        s2 = tf.shape(y_hat)

        labels = tf.reshape(y, shape=[s1[0], s1[2], s1[3], s1[4]])
        predictions = tf.reshape(y_hat, shape=[s2[0], s2[2], s2[3], 1])

        size_image = tf.shape(predictions)[1]
        cropped_predictions = predictions[:, self.border:size_image - self.border, self.border:size_image - self.border, :]
        cropped_labels = labels[:, self.border:size_image - self.border, self.border:size_image - self.border, :]
        cropped_mask_y = self.mask_y[:, :, self.border:size_image - self.border, self.border:size_image - self.border, :]

        cropped_mask_y = tf.squeeze(cropped_mask_y, axis=1)
        cropped_predictions_masked = cropped_predictions * cropped_mask_y
        cropped_labels_masked = cropped_labels * cropped_mask_y

        eps = tf.constant(1e-8, dtype=tf.float32)
        b = (1.0 / (tf.reduce_sum(cropped_mask_y, axis=[1, 2, 3]) + eps)) * tf.reduce_sum(
            cropped_labels_masked - cropped_predictions_masked, axis=[1, 2, 3])
        b = tf.reshape(b, [s1[0], 1, 1, 1])

        corrected_predictions = (cropped_predictions_masked + b) * cropped_mask_y

        max_val = tf.reduce_max(cropped_labels_masked) - tf.reduce_min(cropped_labels_masked) + eps
        ssim = tf.image.ssim(cropped_labels_masked, corrected_predictions, max_val=max_val)
        return tf.reduce_mean(tf.cast(ssim, tf.float32))

    def summary(self):
        with tf.compat.v1.name_scope('performance') as scope:
            loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)
            score_summary = tf.compat.v1.summary.scalar('score', self.score)

        self.summary_loss = tf.compat.v1.summary.merge([loss_summary])
        self.summary_metric = tf.compat.v1.summary.merge([score_summary])

        with tf.compat.v1.name_scope('images') as scope:
            tf.summary.image('images_temp_SR',
                             tf.reshape(self.SR_temp, [-1, tf.shape(self.SR_temp)[2], tf.shape(self.SR_temp)[3], 1]),
                             3)

        with tf.compat.v1.name_scope('filters') as scope:
            filters = tf.reshape(self.filters, [-1, self.dyn_filter_size * 8, self.dyn_filter_size])
            filters = tf.expand_dims(filters, axis=-1)
            tf.summary.image('filters', filters, 3)

        self.tf_images_summaries = tf.compat.v1.summary.merge_all(key='images')

    def train_one_epoch(self, saver, train_writer, test_writer, epoch, step):
        start_time = time.time()
        n_batches = 0
        total_loss = 0

        if self.full:
            self.gen = load_dataset(
                self.data_cfg,
                self.dataset_path,
                self.n_chunks,
                band=self.spectral_band,
                num_images=9,
                how='best'
            )
        else:
            self.gen = load_dataset_best9(
                self.data_cfg,
                self.dataset_path,
                self.n_chunks,
                band=self.spectral_band
            )

        while True:

            val = self.get_data()

            if val == 0:
                print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
                print('Took: {0} seconds'.format(time.time() - start_time))
                return step

            self.batch_training_norm, \
                self.batch_training_mask, \
                self.batch_training_y_norm, \
                self.batch_mask_train_y, \
                self.shifts, \
                self.y_filters_dyn, \
                self.fill_coeff_train = shuffle(self.batch_training_norm,
                                                self.batch_training_mask,
                                                self.batch_training_y_norm,
                                                self.batch_mask_train_y,
                                                self.shifts,
                                                self.y_filters_dyn,
                                                self.fill_coeff_train
                                                )

            for i in range(1, int(self.batch_training_norm.shape[0] / self.batch_size) + 1):

                if (step + 1) % 200 == 0:
                    _, l, summaries = self.sess.run([self.opt, self.loss, self.summary_loss], feed_dict={
                        self.y_filters: self.y_filters_dyn[(i - 1) * self.batch_size:i * self.batch_size],
                        self.y: self.batch_training_y_norm[(i - 1) * self.batch_size:i * self.batch_size],
                        self.x: self.batch_training_norm[(i - 1) * self.batch_size:i * self.batch_size],
                        self.fill_coeff: self.fill_coeff_train[(i - 1) * self.batch_size:i * self.batch_size],
                        self.mask_y: self.batch_mask_train_y[(i - 1) * self.batch_size:i * self.batch_size]
                    }
                                                    )

                    train_writer.add_summary(summaries, global_step=step)
                    train_writer.flush()

                else:
                    _, l = self.sess.run([self.opt, self.loss], feed_dict={
                        self.y: self.batch_training_y_norm[(i - 1) * self.batch_size:i * self.batch_size],
                        self.y_filters: self.y_filters_dyn[(i - 1) * self.batch_size:i * self.batch_size],
                        self.x: self.batch_training_norm[(i - 1) * self.batch_size:i * self.batch_size],
                        self.fill_coeff: self.fill_coeff_train[(i - 1) * self.batch_size:i * self.batch_size],
                        self.mask_y: self.batch_mask_train_y[(i - 1) * self.batch_size:i * self.batch_size]
                        }

                                         )

                if (step + 1) % self.skip_step == 0:
                    saver.save(self.sess, 'checkpoints/' + self.tensorboard_dir + '/' + 'model.ckpt', step)

                    print('Training Loss for a mini batch at step {0}: {1}'.format(step, l))
                    self.eval_once(test_writer, epoch, step)

                total_loss += l
                n_batches += 1
                step += 1

    def train(self, n_epochs):

        safe_mkdir('checkpoints')
        safe_mkdir('checkpoints/' + self.tensorboard_dir)
        train_writer = tf.compat.v1.summary.FileWriter('./graphs/' + self.tensorboard_dir + '/train',
                                                       tf.compat.v1.get_default_graph())
        test_writer = tf.compat.v1.summary.FileWriter('./graphs/' + self.tensorboard_dir + '/test',
                                                      tf.compat.v1.get_default_graph())
        self.sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.tensorboard_dir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            try:
                print(f"Restoring from checkpoint: {ckpt.model_checkpoint_path}")
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            except Exception as e:
                print(f"Error restoring checkpoint: {e}")
                print("Starting training from scratch.")
        else:
            print("No checkpoint found. Starting training from scratch.")

        ''' 
        # Restore pre-trained registration layers
        dir_checkpoints_registr_net = self.RegNet_pretrain_dir
        variables_to_restore = {
            var.name.split(':')[0]: var
            for var in tf.compat.v1.global_variables()
            if var.name.startswith('Rconv')
        }
        saver_regist_net = tf.compat.v1.train.Saver(variables_to_restore)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(dir_checkpoints_registr_net + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            try:
                print(f"Restoring registration network from checkpoint: {ckpt.model_checkpoint_path}")
                saver_regist_net.restore(self.sess, ckpt.model_checkpoint_path)
            except Exception as e:
                print(f"Error restoring registration network checkpoint: {e}")
                print("Skipping registration network restoration.")
        else:
            print("No pre-trained registration network checkpoint found. Skipping restoration.")
'''

        step = self.gstep.eval(session=self.sess)
        for epoch in range(n_epochs):
            step = self.train_one_epoch(saver, train_writer, test_writer, epoch, step)

        return step

        """ #Restore pre-trained registration layers
            dir_checkpoints_registr_net=self.RegNet_pretrain_dir

            #saver_regist_net = tf.compat.v1.train.Saver({
            #                        'Rconv1b/W':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv1b/W:0'),
            #                        'Rconv2b/W':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv2b/W:0'),
            #                        'Rconv3b/W':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv3b/W:0'),
            #                        'Rconv4b/W':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv4b/W:0'),
            #                        'Rconv7_b/W':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv7_b/W:0'),
            #                        'Rconv1b/b':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv1b/b:0'),
            #                        'Rconv2b/b':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv2b/b:0'),
            #                        'Rconv3b/b':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv3b/b:0'),
            #                        'Rconv4b/b':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv4b/b:0'),
            #                        'Rconv7_b/b':tf.compat.v1.get_default_graph().get_tensor_by_name('Rconv7_b/b:0'),

            #                       })                                    

            # Dynamically fetch variable names
            variables_to_restore = {
                var.name.split(':')[0]: var
                for var in tf.compat.v1.global_variables()
                if var.name.startswith('Rconv')
            }

            saver_regist_net = tf.compat.v1.train.Saver(variables_to_restore)                         

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(dir_checkpoints_registr_net+'/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_regist_net.restore(self.sess, ckpt.model_checkpoint_path)

            #Restore pre-trained  upsampling network 
            dir_checkpoints_upsampling_net=self.SISRNet_pretrain_dir


            conv_W={'conv_up_{0}/W'.format(i):tf.compat.v1.get_default_graph().get_tensor_by_name('conv_up_{0}/W:0'.format(i)) for i in range(8)}
            conv_b={'conv_up_{0}/b'.format(i):tf.compat.v1.get_default_graph().get_tensor_by_name('conv_up_{0}/b:0'.format(i)) for i in range(8)}
            in_beta={'inst_norm_up_{0}/beta'.format(i):tf.compat.v1.get_default_graph().get_tensor_by_name('inst_norm_up_{0}/beta:0'.format(i)) for i in range(8)}
            in_gamma={'inst_norm_up_{0}/gamma'.format(i):tf.compat.v1.get_default_graph().get_tensor_by_name('inst_norm_up_{0}/gamma:0'.format(i)) for i in range(8)}


            all_weights={**conv_W,**conv_b,**in_beta,**in_gamma}

            saver_upsampling_net = tf.compat.v1.train.Saver(all_weights)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(dir_checkpoints_upsampling_net+'/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_upsampling_net.restore(self.sess, ckpt.model_checkpoint_path)        


        step = self.gstep.eval(session=self.sess)

        for epoch in range(n_epochs):
            step = self.train_one_epoch(saver, train_writer,test_writer, epoch, step)


        return step
    """

    def eval(self):
        with tf.compat.v1.name_scope('psnr'):
            self.score = self.PSNR(norm=True)
            self.psnr = self.PSNR(norm=False)

        with tf.compat.v1.name_scope('ssim'):
            self.ssim = self.SSIM()

        with tf.compat.v1.name_scope('accuracy'):
            pred_filter = tf.argmax(self.logits_filters, axis=2, output_type=tf.int32)
            true_filter = tf.argmax(self.y_filters, axis=2, output_type=tf.int32)

            correct = tf.equal(pred_filter, true_filter)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            pred_row = pred_filter // self.dyn_filter_size
            pred_col = pred_filter % self.dyn_filter_size
            true_row = true_filter // self.dyn_filter_size
            true_col = true_filter % self.dyn_filter_size

            delta_row = tf.abs(pred_row - true_row)
            delta_col = tf.abs(pred_col - true_col)

            correct_tol1 = tf.logical_and(delta_row <= 1, delta_col <= 1)
            self.accuracy_tol1 = tf.reduce_mean(tf.cast(correct_tol1, tf.float32))
            self.shift_l1 = tf.reduce_mean(tf.cast(delta_row + delta_col, tf.float32))

    def eval_once(self, writer, epoch, step):
        start_time = time.time()

        score_list = []
        psnr_list = []
        ssim_list = []
        loss_list = []
        accuracy_list = []
        accuracy_tol1_list = []
        shift_l1_list = []

        val_batch_size = 1
        for i in range(1, int(self.batch_validation_norm.shape[0] / val_batch_size) + 1):
            score, psnr, ssim, loss, accuracy, accuracy_tol1, shift_l1 = self.sess.run(
                [self.score, self.psnr, self.ssim, self.loss, self.accuracy, self.accuracy_tol1, self.shift_l1],
                feed_dict={
                    self.y_filters: self.y_filters_valid_dyn[
                        (i - 1) * val_batch_size:i * val_batch_size],
                    self.y: self.batch_validation_y_norm[
                        (i - 1) * val_batch_size:i * val_batch_size],
                    self.x: self.batch_validation_norm[
                        (i - 1) * val_batch_size:i * val_batch_size],
                    self.fill_coeff: self.fill_coeff_valid[
                        (i - 1) * val_batch_size:i * val_batch_size],
                    self.mask_y: self.batch_mask_valid_y[
                        (i - 1) * val_batch_size:i * val_batch_size],
                    self.norm_baseline: self.norm_validation[
                        (i - 1) * val_batch_size:i * val_batch_size]
                })
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            score_list.append(score)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            accuracy_tol1_list.append(accuracy_tol1)
            shift_l1_list.append(shift_l1)

        average_score = np.mean(score_list)
        average_psnr = np.mean(psnr_list)
        average_ssim = np.mean(ssim_list)
        average_loss = np.mean(loss_list)
        average_accuracy = np.mean(accuracy_list)
        average_accuracy_tol1 = np.mean(accuracy_tol1_list)
        average_shift_l1 = np.mean(shift_l1_list)

        summary_average_score = tf.compat.v1.Summary()
        summary_average_score.value.add(tag="performance/score", simple_value=average_score)

        summary_average_psnr = tf.compat.v1.Summary()
        summary_average_psnr.value.add(tag="performance/psnr", simple_value=average_psnr)

        summary_average_ssim = tf.compat.v1.Summary()
        summary_average_ssim.value.add(tag="performance/ssim", simple_value=average_ssim)

        summary_average_loss = tf.compat.v1.Summary()
        summary_average_loss.value.add(tag="performance/loss", simple_value=average_loss)

        summary_average_accuracy = tf.compat.v1.Summary()
        summary_average_accuracy.value.add(tag="performance/filter_accuracy_exact", simple_value=average_accuracy)

        summary_average_accuracy_tol1 = tf.compat.v1.Summary()
        summary_average_accuracy_tol1.value.add(tag="performance/filter_accuracy_tol1", simple_value=average_accuracy_tol1)

        summary_average_shift_l1 = tf.compat.v1.Summary()
        summary_average_shift_l1.value.add(tag="performance/filter_shift_l1", simple_value=average_shift_l1)

        writer.add_summary(summary_average_score, global_step=step)
        writer.add_summary(summary_average_psnr, global_step=step)
        writer.add_summary(summary_average_ssim, global_step=step)
        writer.add_summary(summary_average_loss, global_step=step)
        writer.add_summary(summary_average_accuracy, global_step=step)
        writer.add_summary(summary_average_accuracy_tol1, global_step=step)
        writer.add_summary(summary_average_shift_l1, global_step=step)

        writer.flush()

        print('Score on test batch at epoch {0} (lower is better): {1} '.format(epoch, average_score))
        print('cPSNR on test batch at epoch {0}: {1} dB'.format(epoch, average_psnr))
        print('SSIM on test batch at epoch {0}: {1} '.format(epoch, average_ssim))
        print('Filter exact accuracy on test batch at epoch {0}: {1} '.format(epoch, average_accuracy))
        print('Filter ±1 px accuracy on test batch at epoch {0}: {1} '.format(epoch, average_accuracy_tol1))
        print('Filter shift L1 error on test batch at epoch {0}: {1} '.format(epoch, average_shift_l1))

        print('Took: {0} seconds'.format(time.time() - start_time))

    def predict_test(self, test_dir, n_slide):

        nickname = self.data_cfg.nickname
        #numband = self.data_cfg.numband
        hiresl_h = self.data_cfg.hiresl_h
        hiresl_w = self.data_cfg.hiresl_w

        input_images_LR_test = np.load(
            os.path.join(
                test_dir,
                'dataset_{0}_{1}_s_LR_test.npy'.format(self.spectral_band, nickname)
            ),
            allow_pickle=True
        )

        mask_LR_test = np.load(
            os.path.join(
                test_dir,
                'dataset_{0}_mask_{1}_s_LR_test.npy'.format(self.spectral_band, nickname)
            ),
            allow_pickle=True
        )

        shift_LR_test = np.load(
            os.path.join(
                test_dir,
                'shifts_test_{1}_s_{0}.npy'.format(self.spectral_band, nickname)
            ),
            allow_pickle=True
        )

        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/' + self.tensorboard_dir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        indexes = []

        for image_set in mask_LR_test:
            indexes.append(np.argsort(np.sum(np.array(image_set[0:]), axis=(1, 2)))[::-1])

        input_images_LR_test = [image_set[indexes_set] for image_set, indexes_set in zip(input_images_LR_test, indexes)]
        mask_LR_test = np.array([image_set[indexes_set] for image_set, indexes_set in zip(mask_LR_test, indexes)])

        SR_images = np.zeros([len(input_images_LR_test), 1, hiresl_h, hiresl_w, 1])
        for m in range(0, len(input_images_LR_test)):

            imageset = np.array(input_images_LR_test[m])
            imageset_mask = np.array(mask_LR_test[m])
            imageset_shift = np.array(shift_LR_test[m])

            percentage = 0.9
            while True:
                indexes_0 = np.argwhere((np.sum(imageset_mask[0:], axis=(1, 2)) / (hiresl_h * hiresl_w)) > percentage).squeeze(
                    axis=1)
                indexes_0 = indexes_0 if indexes_0.ndim > 0 else np.array([])

                indexes = np.array(list(indexes_0))
                if indexes.size >= 9:
                    imageset = imageset[indexes]
                    imageset_mask = imageset_mask[indexes]
                    imageset_shift = imageset_shift[indexes]

                    break
                else:

                    percentage -= 0.05
                    continue

            len_imageset = np.shape(imageset)[0]

            temporal_dim = 9
            upper_bound = n_slide
            if len_imageset - temporal_dim + 1 > upper_bound:
                size = upper_bound + 1
            else:
                size = len_imageset - temporal_dim + 1

            SR_imageset = np.zeros([size, 1, hiresl_h, hiresl_w, 1])
            for n in range(0, size):

                imageset_9 = imageset[n:n + temporal_dim]
                imageset_9 = np.expand_dims(imageset_9, axis=0)
                imageset_9 = np.expand_dims(imageset_9, axis=-1)

                imageset_9_mask = imageset_mask[n:n + temporal_dim]
                imageset_9_mask = np.expand_dims(imageset_9_mask, axis=0)
                imageset_9_mask = np.expand_dims(imageset_9_mask, axis=-1)

                imageset_9_mask = np.round(imageset_9_mask)
                imageset_9_mask = imageset_9_mask.astype('bool')

                for j in range(imageset_9_mask.shape[1]):
                    shifted_mask = imageset_9_mask[:, j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), imageset_shift[j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape(
                        [1, np.shape(corrected_mask)[0], np.shape(corrected_mask)[1], 1])
                    imageset_9_mask[:, j] = np.round(corrected_mask)

                sh = imageset_9_mask.shape
                fill_coeff_test = np.ones([sh[0], sh[1], sh[1], sh[2], sh[3], sh[4]], dtype='bool')
                for i in range(0, 9):
                    fill_coeff_test[:, :, i] = np.expand_dims(imageset_9_mask[:, i], axis=1)

                for i in range(0, 9):
                    for j in range(i + 1, 9):
                        rows_indexes = [k for k in range(0, 9) if k != (j)]
                        fill_coeff_test[:, rows_indexes, j] = fill_coeff_test[:, rows_indexes, j] * np.expand_dims(
                            1 - imageset_9_mask[:, i], axis=1)

                for i in range(1, 9):
                    fill_coeff_test[:, i, 0:i] = fill_coeff_test[:, i, 0:i] * np.expand_dims(1 - imageset_9_mask[:, i],
                                                                                             axis=1)

                f = np.sum(fill_coeff_test, axis=2)
                fill_coeff_test[:, range(9), range(9), :, :, :] = fill_coeff_test[:, range(9), range(9), :, :, :] + \
                                                                  np.logical_not(f)[:, range(9), :, :, :]


                imageset_9 = (imageset_9 - self.mu) / self.sigma

                SR_image = self.sess.run(self.logits, feed_dict={
                    self.x: imageset_9,
                    self.fill_coeff: fill_coeff_test
                })

                SR_imageset[n] = SR_image

            SR_imageset_registered = np.empty([0, 1, hiresl_h, hiresl_w, 1])  # MF
            for z in range(SR_imageset.shape[0]):
                reference_image = SR_imageset[0]
                shifted_image = SR_imageset[z]

                shift, error, diffphase = phase_cross_correlation(reference_image.squeeze(), shifted_image.squeeze(),
                                                                  upsample_factor=1)
                if (np.abs(shift) > 4).any():
                    print('Skip image...too large shifts')
                    print(shift)
                    continue

                corrected_image = fourier_shift(np.fft.fftn(shifted_image.squeeze()), shift)
                corrected_image = np.fft.ifftn(corrected_image)
                corrected_image = corrected_image.reshape([1, 1, hiresl_h, hiresl_w, 1])  # MF
                SR_imageset_registered = np.append(SR_imageset_registered, corrected_image, axis=0)

            SR_image = np.mean(SR_imageset_registered, axis=0, keepdims=True)

            SR_image = np.real_if_close(SR_image, tol=1e-5)

            SR_images[m] = SR_image

            print('Image number {0}'.format(m))
        SR_images = (SR_images * self.sigma_rescaled) + self.mu

        return SR_images

    def build(self):
        self.inference_FR()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()



















