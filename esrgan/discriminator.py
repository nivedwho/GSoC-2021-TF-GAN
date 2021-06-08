from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan


def _conv2d(net, filters, kernal_size = 3, stride = 1, bias = False, name = 'conv'):
    return tf.layers.conv2d(net, filters, kernal_size, stride, padding = "SAME",
                            use_bias = bias, name=name)

def _leaky_relu(x, alpha = 0.2, name = 'leakyReLU'):
    return tf.nn.leaky_relu(x, alpha, name)

def _batchNorm(net, name = 'bn'):
  return tf.layers.BatchNormalization(name = name)(net)

def _fc(net, units, name):
  return tf.layers.dense(net, units, name)

def _conv_block(net, out_channels, count):
  with tf.variable_scope('block_{}'.format(count)):
    net = _conv2d(net, filters = out_channels, name = 'Conv1')
    net = _batchNorm(net, name = 'batch_norm_1')
    net = _leaky_relu(net, name = 'leaky_relu_1')

    net = _conv2d(net, filters = out_channels, name = 'Conv2')
    net = _batchNorm(net, name = 'batch_norm_2')
    net = _leaky_relu(net, name = 'leaky_relu_2')

    return net

def ESRGAN_D(net, 
            n_filters = 64, 
            inc_filters = 32, 
            channels = 3, 
            kernel_size = 3):
  
  with tf.variable_scope('input'):
    net = _conv2d(net, filter = n_filters, name = 'conv_1')
    net = _leaky_relu(net, name = 'leaky_relu_1') 
    
    net = _conv2d(net, filter = n_filters, kernal_size = 4, stride = 2, name = 'conv_2')
    net = _batchNorm(net, name = 'batch_norm_1')
    net = _leaky_relu(net, name = 'leaky_relu_2')

  with tf.variable_scope('conv_block'):
    net = _conv_block(net, out_channels = n_filters * 2, count = 0)
    net = _conv_block(net, out_channels = n_filters * 4, count = 1)
    net = _conv_block(net, out_channels = n_filters * 8, count = 2)
    net = _conv_block(net, out_channels = n_filters * 8, count = 3)

  with tf.variable_scope('fc'):
    net = tf.layers.flatten(net)
    net = _fc(net, units = 100, name = 'fully_connected_1')
    net = _leaky_relu(net, name = 'leaky_relu_1')
    net = _fc(net, units = 1 , name = 'fully_connected_2')
  
  return net