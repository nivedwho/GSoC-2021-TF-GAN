"""Tests for tfgan.examples.esrgan.data_provider"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import tensorflow as tf
import data_provider
import collections

Params = collections.namedtuple('HParams', ['hr_dimension', 
                                            'scale', 
                                            'batch_size',
                                            'data_dir'])

class DataProviderTest(tf.test.TestCase, absltest.TestCase):
  
  def setUp(self):
    super(DataProviderTest, self).setUp()
    self.HParams = Params(256, 4, 32, '/content/')
    self.dataset = data_provider.get_div2k_data(self.HParams)
    self.mock_lr = tf.random.normal([32, 64, 64, 3])
    self.mock_hr = tf.random.normal([32, 256, 256, 3])

  def test_dataset(self):
    with self.cached_session() as sess:
      self.assertIsInstance(self.dataset, tf.data.Dataset)
      lr_image, hr_image = next(iter(self.dataset))
      sess.run(tf.compat.v1.global_variables_initializer())

      self.assertEqual(type(self.mock_lr), type(lr_image))
      self.assertEqual(self.mock_lr.shape, lr_image.shape)

      self.assertEqual(type(self.mock_hr), type(hr_image))
      self.assertEqual(self.mock_hr.shape, hr_image.shape)


if __name__ == '__main__':
  tf.test.main()
