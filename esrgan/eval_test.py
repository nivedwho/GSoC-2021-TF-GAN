"""Tests for tfgan.examples.esrgan.eval"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import os
import collections

import tensorflow as tf
import eval_lib, networks

HParams = collections.namedtuple('HParams', [
  'num_steps', 'image_dir', 'batch_size', 'num_inception_images',
  'eval_real_images', 'hr_dimension', 'scale', 'trunk_size'])

class EvalTest(tf.test.TestCase):
  def setUp(self):
    self.HParams = HParams(1, '/content/', 
                          2, 2, 
                          True, 256, 
                          4, 11)
    
    d = tf.data.Dataset.from_tensor_slices(tf.random.normal([2, 256, 256, 3]))
    def lr(hr):
      lr = tf.image.resize(hr, [64, 64], method='bicubic')
      return lr, hr

    d = d.map(lr)
    d = d.batch(2)
    self.mock_dataset = d 
    self.generator = networks.generator_network(self.HParams)

  def test_eval(self):
    self.assertIsNone(eval_lib.evaluate(self.HParams, 
                                        self.generator, 
                                        self.mock_dataset))

if __name__ == '__main__':
  tf.test.main()

