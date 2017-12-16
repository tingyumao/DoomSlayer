import cv2
import numpy as np
import random
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


#####################################
#          RL utilities             #
#####################################
def preprocess(frame_image, shape):
    # convert to grayscale
    frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY) # return a 120x160 array
    # crop out the bottom part, eg., [120, 160] ==> [100, 160]
    #H, W = frame_gray.shape
    #H_crop = int(H*5/6.0)
    
    #frame_crop = frame_gray[:H_crop, :]/255.0
    #frame_gray = np.expand_dims(frame_gray, axis=2) # return a 120x160x1 array

    # resize
    h, w = shape
    frame_resize = cv2.resize(frame_image, (w, h)).astype("float32")
    frame_resize /= 255.0
    #frame_resize = np.expand_dims(frame_resize, axis=2) # return a 120x160x1 array
    
    return frame_resize

def e_greedy_select(actions, qs, eps):
    action_onehot = [0.0 for i in range(len(actions))]
    # qs are q-value for each actions at current state
    max_actions = [a for i, a in enumerate(actions) if qs[i]==max(qs)]

    assert len(max_actions) > 0
    
    if random.random() < eps:
        action = random.choice(actions)
        action_idx = actions.index(action)
        action_onehot[action_idx] = 1.0
    else:
        action = random.choice(max_actions)
        action_idx = actions.index(action)
        action_onehot[action_idx] = 1.0
    
    return action, action_onehot

    
