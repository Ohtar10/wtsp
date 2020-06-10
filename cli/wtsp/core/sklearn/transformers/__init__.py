""" sklearn transformers """
import warnings

import tensorflow as tf


def init_tensorflow():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            from tensorflow.compat.v1 import ConfigProto, InteractiveSession
            config = ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            # Necessary to default a session in tensorflow for keras to grab
            session = InteractiveSession(config=config)