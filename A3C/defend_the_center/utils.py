import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal
import os
import csv
import cv2
import itertools
import tensorflow.contrib.slim as slim
        
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
	frame = cv2.cvtColor(np.transpose(frame,(1,2,0)), cv2.COLOR_BGR2GRAY)
	s = cv2.resize(frame,(120,160))
	s = s[10:-10,30:-30]
	s = cv2.resize(s,(84,84))
	s = np.reshape(s,[np.prod(s.shape)]) / 255.0
	return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer