#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from vizdoom import DoomGame
from vizdoom import GameVariable
from project import networks
from project.utils import preprocess, e_greedy_select, preprocess_labelsbuf
from PIL import Image

ACTIONS_NUM = 8
INITIAL_EPS = 1.0
FINAL_EPS = 0.1
GAMMA = 0.99
FRAME_REPEAT = 4
OBSERVE = 3000
REPLAY_MEMORY = 5000
MAX_TRAIN_EPISODE = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
CHECKPOINTS_PATH = '.\\checkpoint\\train'
h, w, channels = 48, 64, 3  # height, width and RGB channels
reward_record = open('reward.txt', 'w')


def train():
    # Set scenarios
    scene_name = "defend_the_line"
    cfg_path = "../example/scenarios/defend_the_line.cfg"

    # Override default game configurations.
    game = DoomGame()
    game.load_config(cfg_path)
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_render_weapon(True)
    game.init()

    # 3 individual actions
    actions = set_actions()

    # Set NN parameters
    net_cfg = dict()
    # kernel width, kernel height, stride, filters
    net_cfg["conv1_cfg"] = [4, 4, 2, 64]
    net_cfg["conv2_cfg"] = [4, 4, 2, 128]
    net_cfg["fc_size1"] = 512
    net_cfg["fc_size2"] = 128
    net_cfg["fc_size3"] = 32
    net_cfg["action_num"] = ACTIONS_NUM

    # Define the placeholder for states
    # State placeholder
    # Game variables placeholder: angle, health
    # Actions placeholder
    # Target value placeholder
    # labels buffer placeholder
    states_ph = tf.placeholder(tf.float32, shape=(None, h, w, channels + 1))
    actions_ph = tf.placeholder(tf.float32, shape=(None, ACTIONS_NUM))
    ys_ph = tf.placeholder(tf.float32, shape=(None,))

    # Define two networks, forward pass and the target Q network.
    print('Defining networks...')
    with tf.variable_scope('q_forward', reuse=tf.AUTO_REUSE):
        q_net = networks.QNetwork(net_cfg)
        q_values = q_net(states_ph)
    with tf.variable_scope('q_target', reuse=tf.AUTO_REUSE):
        target_q_net = networks.QNetwork(net_cfg)
        target_q_values = target_q_net(states_ph)

    # Define loss, train step and saver
    print('Defining vars...')
    q_action = tf.reduce_sum(tf.multiply(q_values, actions_ph), axis=1)
    cost = tf.reduce_mean(tf.square(ys_ph - q_action))
    # tf.summary.scalar('Loss', cost)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    saver = tf.train.Saver()
    # Define in-game variables
    # Replay cache
    # Initial epsilon
    # Reward recorder
    # Batch loss
    replay_cache = deque()
    eps = INITIAL_EPS
    all_episode_reward = []
    batch_loss = None

    # Create session
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)

    # Load old trained model
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print("Successfully loaded: ", checkpoint.model_checkpoint_path)
        except ValueError:
            print("Old model not found.")
    else:
        print('No previous checkpoint exists.')

    # file_writer = tf.summary.FileWriter(CHECKPOINTS_PATH, session.graph)
    global_step = 0
    max_reward = 10
    for e in range(MAX_TRAIN_EPISODE):
        # Starts a new episode
        game.new_episode()
        t = 0  # Reset the time step at the beginning of a game.
        while not game.is_episode_finished():
            # Get current state, extract vars
            current_state = game.get_state()
            # Current screen buffer
            screen_buf = current_state.screen_buffer
            screen_buf = preprocess(screen_buf.transpose(1, 2, 0), (h, w))

            # Current labels buffer
            labels_buf = current_state.labels_buffer
            labels_buf = preprocess_labelsbuf(
                current_state.labels_buffer, (h, w))

            # Combine state buffer and labels buffer for state_ph
            input_states_ph = np.concatenate([screen_buf, labels_buf], axis=2)
            current_store_state = input_states_ph

            current_q = session.run(q_values, feed_dict={
                states_ph: np.expand_dims(input_states_ph, axis=0)})
            current_q = np.squeeze(current_q)

            # Get an action according to eps-greedy policy and make it.
            current_action, current_a_onehot = e_greedy_select(
                actions, current_q, eps)
            current_reward = game.make_action(current_action, FRAME_REPEAT)

            # See if the games is ended
            terminal = game.is_episode_finished()
            next_state = game.get_state()
            next_store_state = np.zeros_like(current_store_state)

            # Look at the next state. For experience replay purposes.
            if not terminal:
                next_frame = preprocess(
                    next_state.screen_buffer.transpose(1, 2, 0), (h, w))
                next_labels = preprocess_labelsbuf(
                    next_state.labels_buffer, (h, w))
                next_store_state = np.concatenate(
                    [next_frame, next_labels], axis=2)

            # Push into replay cache
            replay_cache.append(
                [current_store_state, current_a_onehot, current_reward, next_store_state, terminal])
            if len(replay_cache) > REPLAY_MEMORY:
                replay_cache = deque(sorted(replay_cache, key=lambda x: x[2]))
                for _ in range(REPLAY_MEMORY - OBSERVE):
                    replay_cache.popleft()

            # Once we collect enough data, start the double Q training.
            if len(replay_cache) > OBSERVE:
                batch_data = random.sample(replay_cache, BATCH_SIZE)

                # Extract data from batch data
                batch_state = [x[0] for x in batch_data]
                batch_action = [x[1] for x in batch_data]
                batch_reward = [x[2] for x in batch_data]
                batch_n_state = [x[3] for x in batch_data]
                batch_terminal = [x[4] for x in batch_data]

                q_batch = session.run(q_values, feed_dict={states_ph: batch_n_state})
                target_q_batch = session.run(target_q_values,
                                             feed_dict={states_ph: batch_n_state})

                y_batch = batch_reward + np.invert(batch_terminal).astype(np.float32) * GAMMA * target_q_batch[
                    np.arange(BATCH_SIZE), np.argmax(q_batch, axis=1)]

                batch_loss, _ = session.run([cost, train_step],
                                            feed_dict={states_ph: batch_state,
                                                       actions_ph: batch_action,
                                                       ys_ph: y_batch})
            global_step += 1
            t += 1

        if (e + 1) % 50 == 0:
            eps -= (eps - FINAL_EPS) / (MAX_TRAIN_EPISODE / 50)
            eps = max(eps, FINAL_EPS)
        if (e + 1) % 30 == 0:
            print('Copying model parameters...')
            copy_model_parameters(session)

        print("*" * 30)
        print("Finish {} th episode at {} th time steps.".format(e, t))
        total_reward = game.get_total_reward()
        if total_reward > max_reward:
            print('Saving models...')
            saver.save(session, CHECKPOINTS_PATH, global_step=e + 1)
            max_reward = total_reward
        all_episode_reward.append(game.get_total_reward())
        print("Reward in {} th episode: {}".format(e, all_episode_reward[-1]))
        reward_record.write(str(all_episode_reward[-1]) + ',')
        if batch_loss is not None:
            print("Mini-batch train loss in {} th episode: {}".format(e, batch_loss))
        print("Size of Replay Cache: {}".format(len(replay_cache)))
    game.close()
    reward_record.close()


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
    actions = [shoot, left, right, shoot_left,
               shoot_right, left_right, shoot_left_right, nothing]
    return actions


def copy_model_parameters(sess):
    param1 = [v for v in tf.trainable_variables(
    ) if v.name.startswith('q_forward')]
    param1 = sorted(param1, key=lambda v: v.name)
    param2 = [v for v in tf.trainable_variables(
    ) if v.name.startswith('q_target')]
    param2 = sorted(param2, key=lambda v: v.name)

    update_ops = []
    for v1, v2 in zip(param1, param2):
        temp_op = v2.assign(v1)
        update_ops.append(temp_op)
    sess.run(update_ops)


if __name__ == '__main__':
    train()
