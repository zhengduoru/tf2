import tensorflow as tf
import json
import os
from absl import flags
import sys


# mode
flags.DEFINE_string('mode', '', '')

# dataset
#flags.DEFINE_string('', '', '')

# task config
flags.DEFINE_string('builder_name', 'mnist_builder', '')
flags.DEFINE_string('data_name', 'mnist_dataset', '')
flags.DEFINE_string('model_name', 'mnist', '')

# training config
flags.DEFINE_integer('epochs', 3, '')

# ckpt
flags.DEFINE_integer('steps_saving_ckpt', 1000, '')
flags.DEFINE_integer('max_ckpt_files_to_keep', 3, '')
flags.DEFINE_string('logdir', './tf_ckpts', '')

# metrics
flags.DEFINE_string('metrics', 'acc,mean', '')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
