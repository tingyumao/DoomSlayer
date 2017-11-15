import tensorflow as tf
from networks import *
from utils import *
from collections import deque # for replay cache

import numpy as np
from vizdoom import *
import random
import time

def run():

    # set up basic parameters
    ACTIONS_NUM = 8
    INITIAL_EPS = 1.0
    FINAL_EPS = 0.1
    GAMMA = 0.99
    LAST_FRAME_NUM = 1
    FRAME_REPEAT = 1

    TEST_EPISODE = 20

    # load game
    scene_name = "simple"
    cfg_path = "../../example/scenarios/simpler_basic.cfg"
    game = DoomGame()
    game.load_config(cfg_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()

    # define 8 actions
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot_left = [1, 0, 1]
    shoot_right = [0, 1, 1]
    left_right = [1, 1, 0]
    shoot_left_right = [1, 1, 1]
    nothing = [0, 0, 0]
    actions = [shoot, left, right, shoot_left, shoot_right, left_right, shoot_left_right, nothing]

    # define the placeholders for states, used in one-pass forward
    h, w, channels = 30, 45, 3
    states_ph = tf.placeholder(tf.float32, shape=(None, h, w, channels*LAST_FRAME_NUM))
    actions_ph = tf.placeholder(tf.float32, shape=(None, ACTIONS_NUM))
    ys_ph = tf.placeholder(tf.float32, shape=(None,))

    # define the network
    net_cfg = dict()
    net_cfg["conv1_cfg"] = [8,8,3,8] # kernel width, kernel height, stride, filters
    net_cfg["conv2_cfg"] = [3,3,2,8]
    net_cfg["fc_size"] = 128
    net_cfg["action_num"] = ACTIONS_NUM
    with tf.variable_scope("q_network"):
        net = QNetwork(net_cfg)

    # define q valuess
    q_values = net(states_ph)

    # start a tensorflow session
    # The only difference with a regular Session is that 
    # an InteractiveSession installs itself as the default session 
    # on construction. The methods tf.Tensor.eval and 
    # tf.Operation.run will use that session to run ops.
    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())

    # load model
    checkpoint = tf.train.get_checkpoint_state("saved_networks/"+scene_name)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Failed to load model. Please check if the old model exists")
        return

    # start testing
    for e in range(TEST_EPISODE):
        game.new_episode()
        t = 0 # reset the time step at the beginning of a game.
        while not game.is_episode_finished():
            current_state = game.get_state()
            current_frame = current_state.screen_buffer
            current_frame = preprocess(current_frame.transpose(1,2,0))
            # stack last 4 frames together
            if t == 0:
                # return a 30x45x3 numpy array
                current_store_state = current_frame#np.concatenate([current_frame, current_frame, current_frame, current_frame], axis=2)
            else:
                current_store_state = current_frame#np.concatenate([current_store_state[:,:,-3:], current_frame], axis=2) # 100x160x4
            
            # feed current state into network and get the q value for each action.
            current_q = session.run(q_values, feed_dict={states_ph: np.expand_dims(current_store_state, axis=0)}) # return 1x3
            current_q = np.squeeze(current_q)
            # get best action by greedy policy
            current_best_action = actions[np.argmax(current_q)]
            game.make_action(current_best_action, FRAME_REPEAT)
            # update t
            t += 1
            # sleep few mili-seconds to slow down the video
            time.sleep(0.2)

        print("Game reward: {}".format(game.get_total_reward()))
        time.sleep(2)


if __name__  == "__main__":
    run()
