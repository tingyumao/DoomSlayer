#!/usr/bin/env python3
# coding: utf-8
from agent import *
from network import *
import configs as cfg

import os
import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
from vizdoom import *

import cv2

from random import choice
from time import sleep
from time import time

max_episode_length = 2100
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 3 # Agent can move Left, Right, or Fire
load_model = False


def main_train(tf_configs=None, play=False):
    tf.reset_default_graph()

    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
        
    #Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
        num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(DoomGame(),i,s_size,a_size,trainer,cfg.model_path,global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(cfg.model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)


def main_play(tf_configs=None):
    tf.reset_default_graph()

    with tf.Session(config=tf_configs) as sess:
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        ag = Player(DoomGame(),0,s_size,a_size, trainer,cfg.model_path, global_episodes)

        print('Loading Model...')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(cfg.model_path)
        saver.restore(sess, os.path.join(cfg.model_path, cfg.model_file))
        print('Successfully loaded!')

        ag.work(sess)


if __name__ == '__main__':

    train = cfg.IS_TRAIN
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if train:
        main_train(tf_configs=config)
    else:
        main_play(tf_configs=config)

