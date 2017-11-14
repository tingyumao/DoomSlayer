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
        self.conv1_cfg = config.get("conv1_cfg", [8,8,4,16])
        self.conv2_cfg = config.get("conv2_cfg", [4,4,2,32])
        self.fc_size = config.get("fc_size", 256)
        self.action_num = config.get("action_num", 3)
        # define weight and bias for the last FC layer which is followed by ReLU. 
        #self.fc_w = weight_variable([flatten_size, 256])
        #self.fc_b = bias_variable([256,])
        
        # define weight and bias for the output. 
        #self.fc_w = weight_variable([256, action_num])
        #self.fc_b = bias_variable([action_num,])
    
    def __call__(self, state):
        
        # conv1
        in_channels = tf.shape(state)[2]
        filter_h, filter_w, strides, out_channels = self.conv1_cfg
        h = tf.layers.conv2d(state, out_channels, [filter_h, filter_w],
                             strides=[strides, strides], padding="valid",data_format='channels_last',name="conv1")
        h = tf.nn.relu(h)
        
        # conv2
        in_channels = tf.shape(h)[2]
        filter_h, filter_w, strides, out_channels = self.conv2_cfg
        h = tf.layers.conv2d(h, out_channels, [filter_h, filter_w],
                             strides=[strides, strides], padding="valid",data_format='channels_last',name="conv2")
        h = tf.nn.relu(h)
        
        # flatten
        h = tf.contrib.layers.flatten(h) # input: h[batch_size, h, w, channels], output: [batch_size, k]
        
        # fc layer
        h = tf.contrib.layers.fully_connected(h, self.fc_size) # the default activation function is ReLU.
        
        # output layer
        out = tf.contrib.layers.fully_connected(h, self.action_num, activation_fn=None)
        
        return out
        
        
        
        
        
        
        
        
        
        
        
        
        