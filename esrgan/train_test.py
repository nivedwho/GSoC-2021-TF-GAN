"""Tests for tfgan.examples.esrgan.train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import os
import collections

import tensorflow as tf
import train_lib

mock = tf.compat.v1.test.mock
HParams = collections.namedtuple('HParams', [
    'batch_size', 'scale',
    'model_dir', 'phase_1',
    'phase_2', 'hr_dimension',
    'data_dir', 'print_steps',
    'trunk_size', 'total_steps',
    'decay_steps', 'decay_factor',
    'lr', 'beta_1',
    'beta_2', 'init_lr',
    'loss_type', 'lambda_',
    'eta', 'image_dir'])

class TrainTest(tf.test.TestCase):
  def setUp(self):
    self.HParams = HParams(32, 4, '/content/', 
                          False, False, 256,
                          '/content/', 1, 11, 1, 1, 
                          0.5, 0.0001, 0.5, 0.001, 0.00005, 
                          'L1', 0.001, 0.5, '/content/')
    d = tf.data.Dataset.from_tensor_slices(tf.random.normal([32, 256, 256, 3]))
    def lr(hr):
      lr = tf.image.resize(hr, [64, 64], method='bicubic')
      return lr, hr

    d = d.map(lr)
    d = d.batch(2)
    self.mock_dataset = d 
  
  def test_pretrain_generator(self):
    """ Executes all the processes inside the phase-1 training step, once.
        (takes about 100s)"""
    self.assertIsNone(train_lib.pretrain_generator(self.HParams, self.mock_dataset))
  
  def test_train_generator(self):
    """ Executes the phase-2 training step for a single step, once
        (takes about 220s)"""
    self.assertIsNone(train_lib.train_esrgan(self.HParams, self.mock_dataset))

if __name__ == '__main__':
  tf.test.main()
