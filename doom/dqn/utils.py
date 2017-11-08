import numpy as np
import tensorflow as tf

######################################
#       tensorflow utilities         #
######################################
def weight_variable(shape, name=None):
    """
    Inputs:
    - shape: a list/tuple of integers
    """
    if name == None:
        name = "weights"
    init = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.get_variable(name, initializer=init)

def bias_variable(shape, name=None):
    """
    Inputs:
    - shape: a list/tuple of integers
    """
    if name == None:
        name = "bias" 
    init = tf.zeros(shape, tf.float32)
    return tf.get_variable(name, initializer=init)