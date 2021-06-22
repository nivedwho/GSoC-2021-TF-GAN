import os
import tensorflow as tf
from absl import logging


def getstrategy():
    """
    #Colab TPU
    try:
      tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
      print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
      raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    """

    #GPU - Mirrored Strategy
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    return strategy

def load_checkpoint(checkpoint, training_phase, basepath = "" ):
    dir_ = 'checkpoints/' + training_phase
    if basepath:
        dir_ = os.path.join(basepath, dir_)
    if tf.io.gfile.exists(os.path.join(dir_, "checkpoint")):
        logging.info("Found checkpoint at: %s" % dir_)
        status = checkpoint.restore(tf.train.latest_checkpoint(dir_))
        return status

def save_checkpoint(checkpoint, training_phase, basepath =""):
    dir_ = 'checkpoints/' + training_phase
    if basepath:
        dir_ = os.path.join(basepath, dir_)
    dir_ = os.path.join(dir_, os.path.basename(dir_))

    checkpoint.save(file_prefix = dir_)

def preprocess_input(image):
    image = image[..., ::-1]
    mean = -tf.constant([103.939, 116.779, 123.68])
    return tf.nn.bias_add(image, mean)