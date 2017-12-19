import tensorflow as tf


class QNetwork(object):
    """
    Conv_Net + FC_Net
    
    input: for example [200, 320, 3*4]
    
    Network: This structure is simply based on the one 
    shown in AtariDQN Paper. Since doom's scene is more 
    complicated, maybe we should use more conv layer. Also,
    Batch Normalization(BN) may be useful to reduce the variation.
    Additionally, should we use max-pooling?
    
    Conv[4, 4, 2, 64] ==> (BN) ==> ReLU ==> Conv[4, 4, 2, 128]
    ==> (BN) ==> ReLU ==> Flatten ==> Linear(256) ==> 
    output(size is the same as the action set)
    
    """

    def __init__(self, config):
        self.conv1_cfg = config.get("conv1_cfg", [4, 4, 2, 64])  # kernel width, kernel height, stride, filters
        self.conv2_cfg = config.get("conv2_cfg", [4, 4, 2, 128])
        self.fc_size = config.get("fc_size", 512)
        self.action_num = config.get("action_num", 8)
        self.vars_fc1 = config.get("vars_fc1", 64)
        self.vars_fc2 = config.get("vars_fc2", 128)
        self.depth_fc1 = config.get("depth_fc1", 64)
        self.depth_fc2 = config.get("depth_fc2", 10)

    def __call__(self, state, vars, depth):
        # First we deal with the input state
        # convolution 1
        filter_h, filter_w, strides, out_channels = self.conv1_cfg
        h = tf.layers.conv2d(state, out_channels, [filter_h, filter_w],
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             strides=[strides, strides],
                             padding="valid",
                             data_format='channels_last',
                             name="conv1")
        h = tf.nn.relu(h)
        # convolution 2
        filter_h, filter_w, strides, out_channels = self.conv2_cfg
        h = tf.layers.conv2d(h, out_channels, [filter_h, filter_w],
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             strides=[strides, strides],
                             padding="valid",
                             data_format='channels_last',
                             name="conv2")
        h = tf.nn.relu(h)
        # flatten
        h = tf.contrib.layers.flatten(h)  # input: h[batch_size, h, w, channels], output: [batch_size, k]

        # Next we deal with the in-game variables
        vars_fc1 = tf.contrib.layers.fully_connected(vars, self.vars_fc1)
        vars_fc1 = tf.nn.tanh(vars_fc1)
        vars_fc1 = tf.layers.dropout(vars_fc1, rate=0.3, training=True)
        vars_fc2 = tf.contrib.layers.fully_connected(vars_fc1, self.vars_fc2)
        vars_fc2 = tf.nn.tanh(vars_fc2)
        vars_fc2 = tf.layers.dropout(vars_fc2, rate=0.3, training=True)

        # In addition, use depth buffer information
        filter_h, filter_w, strides, out_channels = 4, 4, 2, 64
        d1 = tf.layers.conv2d(depth, out_channels, [filter_h, filter_w],
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              strides=[strides, strides],
                              padding="valid",
                              data_format='channels_last',
                              name='depth_conv1')
        d1 = tf.nn.relu(d1)

        filter_h, filter_w, strides, out_channels = 4, 4, 2, 128
        d2 = tf.layers.conv2d(d1, out_channels, [filter_h, filter_w],
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              strides=[strides, strides],
                              padding="valid",
                              data_format='channels_last',
                              name='depth_conv2')
        d2 = tf.nn.relu(d2)
        dc1 = tf.layers.flatten(d2)
        dc1 = tf.contrib.layers.fully_connected(dc1, self.depth_fc1)
        dc1 = tf.layers.dropout(dc1, rate=0.3, training=True)
        dc2 = tf.nn.tanh(dc1)
        dc2 = tf.contrib.layers.fully_connected(dc2, self.depth_fc2)
        dc2 = tf.layers.dropout(dc2, rate=0.5, training=True)
        dc2 = tf.nn.tanh(dc2)

        # concatenate flattened vector with health value
        concat = tf.concat([h, dc2, vars_fc2], axis=1)
        # fc layer
        concat = tf.contrib.layers.fully_connected(concat, self.fc_size)
        # output layer
        out = tf.contrib.layers.fully_connected(concat, self.action_num, activation_fn=None)
        return out
