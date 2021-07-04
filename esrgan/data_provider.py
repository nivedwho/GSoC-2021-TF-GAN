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

def hr_data(mode, data_dir):
    """
    Download the HR images from DIV2K dataset and return a tf.data.Dataset 
    consisting of tensors of HR images. 
    """
    target_dir = data_dir + 'HR_{}_data'.format(mode)
    download_hr(mode, target_dir)
    
    start_index = 1 if mode == 'train' else 801
    end_index = 801 if mode == 'train' else 901

    img_list = [target_dir + '/DIV2K_{}_HR/{:04}.png'.format(mode, image_id) 
                for image_id in range(start_index,end_index)]

    ds = tf.data.Dataset.from_tensor_slices(img_list)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    return ds

def lr_data(mode, data_dir):
    """
    Download the LR images from DIV2K dataset and return a tf.data.Dataset 
    consisting of tensors of LR images. 
    """
    target_dir = data_dir + 'LR_{}_data'.format(mode)
    download_lr(mode, target_dir)
    start_index = 1 if mode == 'train' else 801
    end_index = 801 if mode == 'train' else 901
    
    img_list = [target_dir + '/DIV2K_{}_LR_bicubic/X4/{:04}x4.png'.format(mode, image_id) 
                for image_id in range(start_index,end_index)]
    
    ds = tf.data.Dataset.from_tensor_slices(img_list)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    return ds


def get_div2k_data(data_dir = None, 
                   mode = 'train',  
                   augment = False, 
                   HR_size = 256, 
                   batch_size = 32,
                   Scale = 4,
                   repeat_count = None):

    """ Downloads and loads DIV2K dataset.
    Args:
        data_dir : Path to the directory where the dataset will be stored.
        mode : Either 'train' or 'valid'.
        augment : Whether to augment the data.
        HR_size : Height/Width of the High Resolution Image. 
        batch_size : Training batch size 
        Scale : Factor by which super resolution is performed. 
                Dimension of an LR image would be HR_size/Scale.
        repeat_count : Repetition of data while training. 
    
    Returns:
        A tf.data.Dataset with pairs of LR image and HR image tensors.

    Raises: 
         TypeError : If the data directory(data_dir) is not specified.
    """

    ds = tf.data.Dataset.zip((lr_data(mode, data_dir), hr_data(mode, data_dir)))
    ds = ds.map(lambda lr, hr: scale(lr, hr, HR_size, Scale), num_parallel_calls=AUTOTUNE)
    
    if augment:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds