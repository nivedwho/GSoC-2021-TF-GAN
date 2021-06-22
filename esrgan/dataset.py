import os
import numpy as np
from absl import logging
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds


def _get_dataset(HParams):
    d = load_dataset(
                    "div2k",
                    scale_down(
                        method= "bicubic",
                        dimension= HParams.hr_dimension),
                    batch_size= HParams.batch_size,
                    data_dir= HParams.data_dir,
                    augment= True,
                    shuffle= True)
    return d

def scale_down(method="bicubic", dimension=256, size=None, factor=4):
  if not size:
    size = (dimension, dimension)
  size_ = {"size": size}

  def scale_fn(image, *args, **kwargs):
    size = size_["size"]
    high_resolution = image
    
    if not kwargs.get("no_random_crop", None):
      high_resolution = tf.image.random_crop(
          image, [size[0], size[1], image.shape[-1]])

    low_resolution = tf.image.resize(
        high_resolution,
        [size[0] // factor, size[1] // factor],
        method=method)

    low_resolution = tf.clip_by_value(low_resolution, 0, 255)
    high_resolution = tf.clip_by_value(high_resolution, 0, 255)
    return low_resolution, high_resolution
  
  scale_fn.size = size_["size"]
  
  return scale_fn


def augment_image(
        brightness_delta=0.05,
        contrast_factor=[0.7, 1.3],
        saturation=[0.6, 1.6]):
  
  def augment_fn(low_resolution, high_resolution, *args, **kwargs):
    def augment_steps_fn(low_resolution, high_resolution):
      def rotate_fn(low_resolution, high_resolution):
        times = tf.random.uniform(minval=1, maxval=4, dtype=tf.int32, shape=[])
        return (tf.image.rot90(low_resolution, times),
                tf.image.rot90(high_resolution, times))

      low_resolution, high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: rotate_fn(low_resolution, high_resolution),
          lambda: (low_resolution, high_resolution))
      
      def flip_fn(low_resolution, high_resolution):
        return (tf.image.flip_left_right(low_resolution),
                tf.image.flip_left_right(high_resolution))
                
      low_resolution, high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: flip_fn(low_resolution, high_resolution),
          lambda: (low_resolution, high_resolution))

      def brightness_fn(low_resolution, high_resolution):
        delta = tf.random.uniform(minval=0, maxval=brightness_delta, dtype=tf.float32, shape=[])
        return (tf.image.adjust_brightness(low_resolution, delta=delta),
                tf.image.adjust_brightness(high_resolution, delta=delta))

      low_resolution, high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: brightness_fn(low_resolution, high_resolution),
          lambda: (low_resolution, high_resolution))

      def contrast_fn(low_resolution, high_resolution):
        factor = tf.random.uniform(
            minval=contrast_factor[0],
            maxval=contrast_factor[1],
            dtype=tf.float32, shape=[])

        return (tf.image.adjust_contrast(low_resolution, factor),
                tf.image.adjust_contrast(high_resolution, factor))
      
      if contrast_factor:
        low_resolution, high_resolution = tf.cond(
            tf.less_equal(tf.random.uniform([]), 0.5),
            lambda: contrast_fn(low_resolution, high_resolution),
            lambda: (low_resolution, high_resolution))

      def saturation_fn(low_resolution, high_resolution):
        factor = tf.random.uniform(
            minval=saturation[0],
            maxval=saturation[1],
            dtype=tf.float32,
            shape=[])
        
        return (tf.image.adjust_saturation(low_resolution, factor),
               tf.image.adjust_saturation(high_resolution, factor))
      
      if saturation:
        low_resolution, high_resolution = tf.cond(
            tf.less_equal(tf.random.uniform([]), 0.5),
            lambda: saturation_fn(low_resolution, high_resolution),
            lambda: (low_resolution, high_resolution))

      return low_resolution, high_resolution

    return tf.cond(
        tf.less_equal(tf.random.uniform([]), 0.2),
        lambda: (low_resolution, high_resolution),
        partial(augment_steps_fn, low_resolution, high_resolution))

  return augment_fn


def reform_dataset(dataset, types, size, num_elems=None):
  _carrier = {"num_elems": num_elems}

  def generator_fn():
    for idx, data in enumerate(dataset, 1):
      if _carrier["num_elems"]:
        if not idx % _carrier["num_elems"]:
          raise StopIteration
      if data[0].shape[0] >= size[0] and data[0].shape[1] >= size[1]:
        yield data[0], data[1]
      else:
        continue
 
  return tf.data.Dataset.from_generator(
      generator_fn, types, (tf.TensorShape([None, None, 3]), tf.TensorShape(None)))
      
def load_dataset(
        name,
        low_res_map_fn,
        split="train",
        batch_size=None,
        shuffle=True,
        augment=True,
        buffer_size=3 * 32,
        cache_dir="cache/",
        data_dir=None,
        options=None,
        num_elems=65536):
  
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  
  dataset = reform_dataset(
      tfds.load(
          name,
          data_dir=data_dir,
          split=split,
          as_supervised=True),
      (tf.float32, tf.float32),
      size=low_res_map_fn.size,
      num_elems=num_elems)
  
  if options:
    dataset = dataset.with_options(options)
  
  dataset = dataset.map(
      low_res_map_fn,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  if batch_size:
    dataset = dataset.batch(batch_size)
  
  dataset = dataset.prefetch(buffer_size)

  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

  if augment:
    dataset = dataset.map(
        augment_image(saturation=None),num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  return dataset
