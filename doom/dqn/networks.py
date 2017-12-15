import tensorflow as tf
from utils import *

class QNetwork(object):
    """
    Conv_Net + FC_Net
    
    input: for example [200, 320, 3*4]
    
    Network: This structure is simply based on the one 
    shown in AtariDQN Paper. Since doom's scene is more 
    complicated, maybe we should use more conv layer. Also,
    Batch Normalization(BN) may be useful to reduce the variation.
    Additionally, should we use max-pooling?
    
    Conv[8,8,4,16] ==> (BN) ==> ReLU ==> Conv[4,4,2,32]
    ==> (BN) ==> ReLU ==> Flatten ==> Linear(256) ==> 
    output(size is the same as the action set)
    
    """
    def __init__(self, config):
        self.conv1_cfg = config.get("conv1_cfg", [8,8,4,16]) # kernel width, kernel height, stride, filters
        self.conv2_cfg = config.get("conv2_cfg", [4,4,2,32])
        self.conv3_cfg = config.get("conv3_cfg", [4,4,2,32])
        self.fc_img = config.get("fc_img", 128)
        self.fc_var = config.get("fc_var", 128)
        self.fc_size = config.get("fc_size", 256)
        self.action_num = config.get("action_num", 3)
        # define weight and bias for the last FC layer which is followed by ReLU. 
        #self.fc_w = weight_variable([flatten_size, 256])
        #self.fc_b = bias_variable([256,])
        
        # define weight and bias for the output. 
        #self.fc_w = weight_variable([256, action_num])
        #self.fc_b = bias_variable([action_num,])
    
    def __call__(self, state, var):
        
        # conv1
        filter_h, filter_w, strides, out_channels = self.conv1_cfg
        h = tf.layers.conv2d(state, out_channels, [filter_h, filter_w],strides=(strides, strides),
                             padding="valid",data_format='channels_last',name="conv1")
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling2d(h, 2, 1)
        
        # conv2
        filter_h, filter_w, strides, out_channels = self.conv2_cfg
        h = tf.layers.conv2d(h, out_channels, [filter_h, filter_w],strides=(strides, strides),
                             padding="valid",data_format='channels_last',name="conv2")
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling2d(h, 2, 1)

        # conv3
        filter_h, filter_w, strides, out_channels = self.conv3_cfg
        h = tf.layers.conv2d(h, out_channels, [filter_h, filter_w],strides=(strides, strides),
                             padding="valid",data_format='channels_last',name="conv3")
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling2d(h, 2, 1)
        
        # flatten
        h = tf.contrib.layers.flatten(h) # input: h[batch_size, h, w, channels], output: [batch_size, k]

        # img fc layer
        h_img = tf.contrib.layers.fully_connected(h, self.fc_img) # the default activation function is ReLU.

        # var fc layer
        h_var = tf.contrib.layers.fully_connected(var, self.fc_var) # the default activation function is ReLU.
        
        # fc layer
        h_total = tf.concat([h_img, h_var], 1)
        h_total = tf.contrib.layers.fully_connected(h_total, self.fc_size) # the default activation function is ReLU.
        
        # output layer
        out = tf.contrib.layers.fully_connected(h_total, self.action_num)
        
        return out
        
        
        
        
        
        
        
        
        
        
        
        
        
