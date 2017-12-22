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
        # kernel width, kernel height, stride, filters
        self.conv1_cfg = config.get("conv1_cfg", [4, 4, 2, 64])
        self.conv2_cfg = config.get("conv2_cfg", [4, 4, 2, 128])
        self.fc_size1 = config.get("fc_size1", 512)
        self.fc_size2 = config.get("fc_size2", 128)
        self.fc_size3 = config.get("fc_size3", 32)
        self.action_num = config.get("action_num", 8)

    def __call__(self, state):
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
        # input: h[batch_size, h, w, channels], output: [batch_size, k]
        h = tf.contrib.layers.flatten(h)

        dc1 = tf.contrib.layers.fully_connected(h, self.fc_size1, activation_fn=tf.nn.relu)
        dc1 = tf.nn.dropout(dc1, keep_prob=0.5)

        dc2 = tf.contrib.layers.fully_connected(dc1, self.fc_size2, activation_fn=tf.nn.relu)
        dc2 = tf.nn.dropout(dc2, keep_prob=0.5)

        dc3 = tf.contrib.layers.fully_connected(dc2, self.fc_size3, activation_fn=tf.nn.relu)
        dc3 = tf.nn.dropout(dc3, keep_prob=0.5)

        # output layer
        out = tf.contrib.layers.fully_connected(
            dc3, self.action_num, activation_fn=None)
        out = tf.sigmoid(out)
        return out
