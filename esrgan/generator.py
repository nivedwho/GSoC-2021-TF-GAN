from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def _leaky_relu(x, alpha = 0.2, name = 'leakyReLU'):
    return tf.nn.leaky_relu(x, alpha, name)

def _conv2d(net, filters, kernal_size = 3, stride = 1, name = 'conv'):
    return tf.layers.conv2d(net, filters, kernal_size, stride, padding = "SAME", name=name)

def _conv2d_transpose(net, filters, kernel_size = 3, stride = 1, name = 'conv_transpose'):
    return tf.layers.conv2d_transpose(net, filters, kernel_size, stride, name)

def _conv_block(net, out_channel, activation=True, count = None):
    with tf.variable_scope('block_{}'.format(count)):
        h = _conv2d(net, out_channel)
        if activation:
            h = _leaky_relu(h)
    return h

def _upsampling_layer(net, num_filter, kernel_size, stride, count):
    h = _conv2d_transpose(net, filters = num_filter, kernel_size = kernel_size, stride = stride, name = 'upsampling_{}'.format(count)) 
    h = _leaky_relu(h)
    return h

def _dense_block(net, inc_filter, beta = 0.2):
    h1 = _conv_block(net, inc_filter, count = 1)
    h2 = tf.concat([net,h1], axis = 3)
    h2 = _conv_block(h2, inc_filter, count = 2)
    h3 = tf.concat([net,h1,h2], axis = 3)
    h3 = _conv_block(h3, inc_filter, count = 3)
    h4 = tf.concat([net,h1,h2,h3], axis = 3)
    h4 = _conv_block(h4, inc_filter, count = 4)
    h5 = tf.concat([net,h1,h2,h3,h4], axis = 3)
    h5 = _conv_block(h5, inc_filter, activaton=False, count = 5)
    return h5 * beta #+ net


def _RRDB(net, in_layer = 64, hidden = 32, beta = 0.2, num = None):
    with tf.variable_scope("RRDB_{}".format(num)):
        with tf.variable_scope('block1'):
            db = _dense_block(net)
        with tf.variable_scope('block2'):
            db = _dense_block(db)
        with tf.variable_scope('block3'):
            out = _dense_block(db)
    return out * beta #+ net


def ESRGAN_G(net, 
             n_filters = 64, 
             inc_filters = 32,
             n_RRDB = 11,
             channels = 3, 
             kernel_size = 3):

    end_points = {}

    with tf.variable_scope('input'):
        net = _conv2d(net, filters = n_filters, name = 'first_conv')
        net = _leaky_relu(net)
        end_points['encoder'] = net 

    net_temp = net
    
    with tf.variable_scope('RRDB'):
        for block_id in xrange(n_RRDB):
            net_temp = _RRDB(net_temp, num = block_id)
            end_points['RRDB_block_%d' % block_id] = net_temp
        
        net_temp = _conv2d(net_temp, filters = n_filters, name = 'rrdb_conv')
        net += net_temp

    with tf.variable_scope('Upsampling'):
        net = _upsampling_layer(net, num_filter = n_filters, stride = 2, count = 1)
        net = _upsampling_layer(net, num_filter = n_filters, stride = 2, count = 2)
        end_points['upsampling'] = net

    with tf.variable_scope('output'):
        net = _conv2d(net, filters = n_filters)
        net = _leaky_relu(net)
        net = _conv2d(net, filters = channels) 

        y = tf.nn.tanh(net)
        end_points['predictions'] = y
    
    return end_points['predictions'], end_points