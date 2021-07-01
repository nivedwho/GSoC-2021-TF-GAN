import os
import tensorflow as tf
import tensorflow_gan as tfgan
from absl import logging

# Utility functions for data processing
def scale(lr_img, hr_img, hr_crop_size, scale):
    """Crops each HR image to hr_crops_size and LR image to hr_crop_size/scale
    
    Args : 
        lr_img : Tensors representing LR images present in the DIV2K dataset.
        hr_img : Tensors representing HR images present in the DIV2K dataset.
        hr_crop_size : Size to which HR images are resized. 
        scale : Integer representing the upsampling factor used to calculate LR image size. 
    
    Returns : 
        Uniformly scaled tensors of LR and HR images present in the dataset. 
    """
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_scaled = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_scaled = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_scaled, hr_img_scaled

def random_flip(lr_img, hr_img):
    """Randomly flips LR and HR images for data augmentation."""
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))

def random_rotate(lr_img, hr_img):
    """Randomly rotates LR and HR images for data augmentation."""
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)

def preprocess_input(image):
    """Preprocessing of images done before calculating loss functions during training."""
    image = image[..., ::-1]
    mean = -tf.constant([103.939, 116.779, 123.68])
    return tf.nn.bias_add(image, mean)

# Utility functions for evaluation
def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
  """Get Frechet Inception Distance between real and generated images.
  
  Args:
    real_images: Real images minibatch.
    generated_images: Generated images minibatch.
    batch_size: Batch dimension.
    num_inception_images: Number of images to run through Inception at once.
  
  Returns:
    Frechet Inception distance. A floating-point scalar.
  
  Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
  """
  # Validate input dimensions.
  real_images.shape[0:1].assert_is_compatible_with([batch_size])
  generated_images.shape[0:1].assert_is_compatible_with([batch_size])

  # Resize input images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_real_images = tf.image.resize(
      real_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  resized_generated_images = tf.image.resize(
      generated_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Compute Frechet Inception Distance.
  num_batches = batch_size // num_inception_images
  fid = tfgan.eval.frechet_inception_distance(
      resized_real_images, resized_generated_images, num_batches=num_batches)

  return fid


def get_inception_scores(images, batch_size, num_inception_images):
    """ Calculate Inception score for images. 

    Args:
        images : A batch of tensors representing series of images.
        batch_size : Integer representing batch dimension.
        num_inception_images : Number of images to run through Inception at once.
    
    Returns:
        Inception scores : A tensor consisting of scores for each batch of images. 
    
    Raises:
        ValueError: If `batch_size` is incompatible with the first dimension of
        `images`.
        ValueError: If `batch_size` isn't divisible by `num_inception_images`.
    """

    # Validate inputs.
    images.shape[0:1].assert_is_compatible_with([batch_size])
    if batch_size % num_inception_images != 0:
        raise ValueError('`batch_size` must be divisible by `num_inception_images`.')
    
    # Resize images.
    size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    resized_images = tf.image.resize(
        images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

    # Run images through Inception.
    num_batches = batch_size // num_inception_images
    inc_score = tfgan.eval.inception_score(
        resized_images, num_batches=num_batches)

    return inc_score