import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functools import partial
import argparse
from absl import flags
import train, model, dataset
from tensorflow.python.eager import profiler
import tensorflow as tf

import logging


logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))


flags.DEFINE_integer('batch_size', 32,
                    'Batch size')

flags.DEFINE_string('model_dir', None,
                    'directory to put the model in.')

flags.DEFINE_integer('hr_dimension', 512, 'Dimension of HR image')

flags.DEFINE_string('data_dir', None, 'directory to put the data.')
flags.DEFINE_boolean('manual', False, 'specify if data_dir is a manual directory.')

flags.DEFINE_string('strategy', None, 'lol')

flags.DEFINE_integer('total_steps', 400000 , 'lol')

flags.DEFINE_integer('decay_steps', 200000 , 'lol')

flags.DEFINE_float('decay_factor', 0.5, 'lol')

flags.DEFINE_float('beta_1', 0.9 , 'lol')

flags.DEFINE_float('beta_2', 0.999, 'Batch size')

flags.DEFINE_float('init_lr', 0.0002 , 'Batch size')

flags.DEFINE_string('loss_type', "L1" , 'Batch size') 
flags.DEFINE_float('lambda_', 0.005 , 'Batch size')
flags.DEFINE_float('eta', 0.01 , 'Batch size')

flags.DEFINE_string('checkpoint_path', None ,'Batch size')                 

FLAGS = flags.FLAGS

def main():
    hparams = train.HParams(
        16, '/content/directory', 512, None, False, None,
        400000, 200000, 0.5, 0.9,  0.999,
        0.0002,"L1", 0.005, 0.01, None)
    
    """
    hparams = train.HParams(
        FLAGS.batch_size, FLAGS.model_dir, FLAGS.hr_dimension, FLAGS.data_dir, FLAGS.strategy,
        FLAGS.total_steps, FLAGS.decay_steps, FLAGS.decay_factor, FLAGS.beta_1, FLAGS.beta_2,
        FLAGS.init_lr, FLAGS.loss_type, FLAGS.lambda_, FLAGS.eta, FLAGS.checkpoint_path)
    """
    
    data = dataset._get_dataset(hparams)
    train.pretrain_generator(hparams, data)
    train.train_esrgan(hparams, data)

if __name__ == '__main__':
    main()





