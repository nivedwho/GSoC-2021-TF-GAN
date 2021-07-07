import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE
from utils import random_rotate, random_flip, scale

def download_hr(mode, target_dir):
    filename = 'DIV2K_{}_HR.zip'.format(mode)
    os.makedirs(target_dir, exist_ok=True)
    source_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/' + filename 
    tf.keras.utils.get_file(filename, source_url, cache_subdir=target_dir, extract = True)

def download_lr(mode, target_dir):
    filename = 'DIV2K_{}_LR_bicubic_X4.zip'.format(mode)
    os.makedirs(target_dir, exist_ok=True)
    source_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/' + filename
    tf.keras.utils.get_file(filename, source_url, cache_subdir=target_dir, extract = True)

def hr_data(mode, data_dir, download_hr_data):
    """
    Download the HR images from DIV2K dataset and return a tf.data.Dataset 
    consisting of tensors of HR images. 
    """
    target_dir = data_dir + 'HR_{}_data'.format(mode)
    
    # Option to skip downloading if already downloaded the images.
    if download_hr_data:
        download_hr(mode, target_dir)
    
    start_index = 1 if mode == 'train' else 801
    end_index = 801 if mode == 'train' else 901

    img_list = [target_dir + '/DIV2K_{}_HR/{:04}.png'.format(mode, image_id) 
                for image_id in range(start_index,end_index)]

    ds = tf.data.Dataset.from_tensor_slices(img_list)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    return ds

def lr_data(mode, data_dir, download_lr_data):
    """
    Download the LR images from DIV2K dataset and return a tf.data.Dataset 
    consisting of tensors of LR images. 
    """
    target_dir = data_dir + 'LR_{}_data'.format(mode)

    # Option to skip downloading if already downloaded the images.
    if download_lr_data: 
        download_lr(mode, target_dir)
    
    start_index = 1 if mode == 'train' else 801
    end_index = 801 if mode == 'train' else 901
    
    img_list = [target_dir + '/DIV2K_{}_LR_bicubic/X4/{:04}x4.png'.format(mode, image_id) 
                for image_id in range(start_index,end_index)]
    
    ds = tf.data.Dataset.from_tensor_slices(img_list)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    return ds


def get_div2k_data(HParams,
                   mode = 'train',  
                   download_hr_data = True,
                   download_lr_data = True,
                   augment = False, 
                   repeat_count = None):

    """ Downloads and loads DIV2K dataset.
    Args:
        HParams : For getting values for parameters such as data directory, batch size etc.
        mode : Either 'train' or 'valid'.
        download_hr : Whether to download the DIV2K dataset HR images. 
        download_lr : Whether to download the DIV2K dataset LR images. 
        augment : Whether to augment the data.
        repeat_count : Repetition of data while training. 
    
    Returns:
        A tf.data.Dataset with pairs of LR image and HR image tensors.

    Raises: 
         TypeError : If the data directory(data_dir) is not specified.
    """

    ds = tf.data.Dataset.zip((lr_data(mode, HParams.data_dir, download_lr), hr_data(mode, HParams.data_dir, download_hr)))
    ds = ds.map(lambda lr, hr: scale(lr, hr, HParams.hr_dimension, HParams.scale), num_parallel_calls=AUTOTUNE)
    
    if augment:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(HParams.batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
