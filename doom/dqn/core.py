import numpy as np
import tensorflow as tf
from networks import *

class ReplayCache(object):
    def __init__(self):
        self.transitions = []
        self.max_space = 1000000 # one-million here, define the maximum number for storing, not sure if it is appropriate.
        
    def add_transition(state, action, reward, next_state):
        transitions, max_space = self.transitions, self.max_space
        
        non_terminal = 1.0
        if next_state == None:
            next_state = -1*np.ones_like(current_state)
            non_terminal = 0.0
            
        transitions.append([state, action, reward, next_state, non_terminal])
        if len(transitions) > max_space:
            transitions = transitions[-max_space:]

            
def model(state_size, train_cfg):
    
    net_cfg = dict()
    net_cfg["conv1_cfg"] = [8,8,4,16]
    net_cfg["conv2_cfg"] = [4,4,2,32]
    net_cfg["fc_size"] = 256
    net_cfg["action_num"] = 3
    
    # discounted factor
    gamma = 0.9
    
    # define placeholder for current_states and next_states, a minibatch-data extracted from Replay cache
    h, w, channels = state_size
    # phi_j in paper, state value is from 0 to 1, or 0 to 255?
    current_states = tf.placeholder(tf.float32, shape=(None, h, w, channels))
    # action_j
    current_actions = tf.placeholder(tf.int32, shape=(None,))
    # phi_{j+1} in paper and terminal state is set as a tensor with all -1s.
    next_states = tf.placeholder(tf.float32, shape=(None, h, w, channels))
    # reward_j
    rewards = tf.placeholder(tf.float32, shape=(None,))
    # non_terminal: each item inside implies whether the current_state is a terminal state or not.
    # If it is a terminal state, then the value will be 0. Otherwise, it will be 1.
    non_terminal = tf.placeholder(tf.float32, shape=(None,))
    
    # define network
    with tf.variable_scope("q_network"):
        net = QNetwork(net_cfg)
        
    # forward
    q_current = net(current_states) # batch_size * action_num
    q_next = net(next_states) # batch_size * action_num
    
    # get y_j
    max_q_a_prime = tf.reduce_max(q_next, axis=1) # batch_size,
    y = rewards + gamma*non_terminal*max_q_a_prime # batch_size,
    
    # loss
    batch_idxs = tf.range(tf.shape(current_states)[0])
    current_action_idxs = tf.stack([batch_idxs, current_actions], axis=1)
    q_a_current = tf.gather_nd(q_current, current_action_idxs) # batch_size, q_a_current = q_current[:, current_actions]
    
    mse_loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - q_a_current))
    
    # back propagation
    params = tf.trainable_variables()
    grads = tf.gradients(mse_loss, params)
    grads, _ = tf.clip_by_global_norm(grads, train_cfg["max_grad_norm"])

    # define optimizer
    lr_init = train_cfg["lr_init"]
    lr_min = train_cfg["lr_min"]
    # not sure if we need to decay learning rate or not.
    decay_steps = train_cfg["decay_step"]
    decay_rate = train_cfg["decay_rate"]
    ## Optional Variable to increment by one after the variables have been updated.
    global_step = tf.get_variable("global_step", initializer=0, trainable=False)
    lr = tf.train.exponential_decay(lr_init, global_step, decay_steps, decay_rate, staircase=True)
    lr = tf.maximum(lr, lr_min)
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.apply_gradients(zip(grads, params), global_step=global_step)
    
    outputs = [current_states, current_actions, next_states, rewards, non_terminal, mse_loss, lr, train_step]
    
    return outputs


    
    
    
    
    
    
    
    