#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import os
from collections import deque
from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
from vizdoom import Mode
from ddqn import networks
from ddqn.utils import preprocess, e_greedy_select

ACTIONS_NUM = 8
INITIAL_EPS = 1.0
FINAL_EPS = 0.1
GAMMA = 0.99
LAST_FRAME_NUM = 1
FRAME_REPEAT = 8
OBSERVE = 3000
REPLAY_MEMORY = 10000
MAX_TRAIN_EPISODE = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
CHECKPOINTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '\\checkpoint'
if not os.path.exists(CHECKPOINTS_PATH):
    os.mkdir(CHECKPOINTS_PATH)


def train():
    # Set scenarios
    scene_name = "defend_the_line"
    cfg_path = "../example/scenarios/defend_the_line.cfg"

    game = DoomGame()
    # Configure game. Other configurations can be found at the .cfg file. The following override.
    game.load_config(cfg_path)
    # game.set_mode(Mode.SPECTATOR)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(False)

    # Initialize the game window
    game.init()

    # 3 individual actions make up 8 possible combinations
    actions = set_actions()

    # Set NN parameters
    net_cfg = dict()
    net_cfg["conv1_cfg"] = [4, 4, 2, 64]  # kernel width, kernel height, stride, filters
    net_cfg["conv2_cfg"] = [4, 4, 2, 128]
    net_cfg["branch_fc1"] = 32
    net_cfg["branch_fc2"] = 64
    net_cfg["fc_size"] = 512
    net_cfg["action_num"] = ACTIONS_NUM

    # Define the placeholder for states
    h, w, channels = 64, 48, 3  # height, width and RGB channels
    # Stack LAST_FRAME_NUM frame together for training. Sometimes no luan use.
    states_ph = tf.placeholder(tf.float32, shape=(None, h, w, channels * LAST_FRAME_NUM))
    # To store information about current health. Could be useful in the 'defend the line' scenario.
    vars_ph = tf.placeholder(tf.float32, shape=(None, 2))
    # Actions placeholder. Represented to be one hot vectors
    actions_ph = tf.placeholder(tf.float32, shape=(None, ACTIONS_NUM))
    # Ground truth. The max Q-values of the next state.
    ys_ph = tf.placeholder(tf.float32, shape=(None,))

    # Define two networks, forward pass and the target Q network.
    with tf.variable_scope('q_forward', reuse=tf.AUTO_REUSE):
        q_net = networks.QNetwork(net_cfg)
        q_values = q_net(states_ph, vars_ph)
    with tf.variable_scope('q_target', reuse=tf.AUTO_REUSE):
        target_q_net = networks.QNetwork(net_cfg)
        target_q_values = target_q_net(states_ph, vars_ph)

    # Define loss
    q_action = tf.reduce_sum(tf.multiply(q_values, actions_ph), axis=1)
    cost = tf.reduce_mean(tf.square(ys_ph - q_action))

    # Define train step
    # Use RMSProp optimizer to minimize cost
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cost)

    # Define replay cache
    replay_cache = deque()

    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)
    # Load old trained model
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print("Successfully loaded: ", checkpoint.model_checkpoint_path)
        except ValueError:
            print("Old model not found.")

    eps = INITIAL_EPS
    all_episode_reward = []  # record total reward in each episode.
    batch_loss = None
    for e in range(MAX_TRAIN_EPISODE):
        # Starts a new episode
        game.new_episode()
        t = 0  # Reset the time step at the beginning of a game.
        while not game.is_episode_finished():
            # Get current state
            current_state = game.get_state()
            current_angle = game.get_game_variable(GameVariable.ANGLE)
            current_health = game.get_game_variable(GameVariable.HEALTH)

            current_vars = np.asarray([current_angle / 360., current_health / 100.])
            screen_buf = current_state.screen_buffer
            # Pre-process from 640×480 to h×w size
            screen_buf = preprocess(screen_buf.transpose(1, 2, 0), (h, w))
            if t == 0:
                current_store_state = screen_buf
            else:
                current_store_state = screen_buf

            current_q = session.run(q_values, feed_dict={states_ph: np.expand_dims(current_store_state, axis=0),
                                                         vars_ph: np.expand_dims(current_vars, axis=0)})
            current_q = np.squeeze(current_q)

            # Get an action according to eps-greedy policy and make it.
            current_action, current_a_onehot = e_greedy_select(actions, current_q, eps)
            current_reward = game.make_action(current_action, FRAME_REPEAT)

            # See if the games is ended
            terminal = game.is_episode_finished()
            next_state = game.get_state()
            next_store_state = np.zeros_like(current_store_state)

            # Look at the next state. For experience replay purposes.
            if not terminal:
                next_frame = preprocess(next_state.screen_buffer.transpose(1, 2, 0), (h, w))
                next_store_state = next_frame

            # Push into replay cache
            replay_cache.append(
                [current_store_state, current_a_onehot, current_reward, next_store_state, current_vars, terminal])
            if len(replay_cache) > REPLAY_MEMORY:
                replay_cache.popleft()

            # Once we collect enough data, start the double Q training.
            if len(replay_cache) > OBSERVE:
                batch_data = random.sample(replay_cache, BATCH_SIZE)

                # Extract data from batch data
                batch_state = [x[0] for x in batch_data]
                batch_action = [x[1] for x in batch_data]
                batch_reward = [x[2] for x in batch_data]
                batch_n_state = [x[3] for x in batch_data]
                batch_vars = [x[4] for x in batch_data]
                batch_terminal = [x[5] for x in batch_data]

                # Additional process for health
                # batch_health = np.expand_dims(batch_health, axis=-1)
                # y_batch = []
                q_batch = session.run(q_values, feed_dict={states_ph: batch_n_state, vars_ph: batch_vars})
                target_q_batch = session.run(target_q_values,
                                             feed_dict={states_ph: batch_n_state, vars_ph: batch_vars})

                # for i in range(BATCH_SIZE):
                #     if batch_terminal[i]:
                #         y_batch.append(batch_reward[i])
                #     else:
                #         y_batch.append(batch_reward[i] + GAMMA * np.max(q_batch))
                y_batch = batch_reward + np.invert(batch_terminal).astype(np.float32) * GAMMA * target_q_batch[
                    np.arange(BATCH_SIZE), np.argmax(q_batch, axis=1)]

                batch_loss, _ = session.run([cost, train_step],
                                            feed_dict={states_ph: batch_state, actions_ph: batch_action,
                                                       ys_ph: y_batch, vars_ph: batch_vars})
            t += 1

            if (e + 1) % 100 == 0:
                eps -= (eps - FINAL_EPS) / (MAX_TRAIN_EPISODE / 100)
                eps = max(eps, FINAL_EPS)
            if (e + 1) % 100 == 0:
                print('Saving models...')
                copy_model_parameters(session)
                saver.save(session, CHECKPOINTS_PATH, global_step=e + 1)
        print("*" * 30)
        print("Finish {} th episode at {} th time steps.".format(e, t))
        all_episode_reward.append(game.get_total_reward())
        print("Reward in {} th episode: {}".format(e, all_episode_reward[-1]))
        if batch_loss is not None:
            print("Minibatch train loss in {} th episode: {}".format(e, batch_loss))
        print("Size of Replay Cache: {}".format(len(replay_cache)))
        print("*" * 30)
    game.close()


def set_actions():
    """
    Just move part of the code from train() to make it look neat.
    :return: list of actions
    """
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot_left = [1, 0, 1]
    shoot_right = [0, 1, 1]
    left_right = [1, 1, 0]
    shoot_left_right = [1, 1, 1]
    nothing = [0, 0, 0]
    actions = [shoot, left, right, shoot_left, shoot_right, left_right, shoot_left_right, nothing]
    return actions


def copy_model_parameters(sess):
    param1 = [v for v in tf.trainable_variables() if v.name.startswith('q_forward')]
    param1 = sorted(param1, key=lambda v: v.name)
    param2 = [v for v in tf.trainable_variables() if v.name.startswith('q_target')]
    param2 = sorted(param2, key=lambda v: v.name)

    update_ops = []
    for v1, v2 in zip(param1, param2):
        temp_op = v2.assign(v1)
        update_ops.append(temp_op)
    sess.run(update_ops)


if __name__ == '__main__':
    train()
