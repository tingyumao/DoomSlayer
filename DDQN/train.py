import tensorflow as tf
from networks import *
from utils import *
from collections import deque # for replay cache

import numpy as np
from vizdoom import *
import random
import time

def train():

    # load game
    scene_name = "roket-basic"
    cfg_path = "../../example/scenarios/rocket_basic.cfg"
    game = DoomGame()
    game.load_config(cfg_path)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    
    # set up basic parameters
    ACTIONS_NUM = 8
    INITIAL_EPS = 1.0
    FINAL_EPS = 0.1
    GAMMA = 0.99
    LAST_FRAME_NUM = 1
    FRAME_REPEAT = 12
    #MAX_TRAIN_STEP = 3000000
    OBSERVE = 3000
    REPLAY_MEMORY = 10000
    MAX_TRAIN_EPISODE = 10000

    LEARNING_RATE = 0.001
    BATCH_SIZE = 64

    assert REPLAY_MEMORY <= 25000
    
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
    
    # initialize state
    #state = game.get_state()
    #img = state.screen_buffer # return a numpy array 3x120x160
    #img = preprocess(img) # 100x160
    #current_state = np.stack([img, img, img, img], axis=2)
    
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
    
    # define loss
    q_values = net(states_ph)
    q_s_a = tf.reduce_sum(q_values * actions_ph, axis=1)
    loss = tf.reduce_mean(tf.square(ys_ph - q_s_a))
    
    # define train_step
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # define replay cache
    replay_cache = deque()
    
    # start a tensorflow session
    # The only difference with a regular Session is that 
    # an InteractiveSession installs itself as the default session 
    # on construction. The methods tf.Tensor.eval and 
    # tf.Operation.run will use that session to run ops.
    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/"+scene_name)
    # load old trained model
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Not found old model")
    
    # start training
    eps = INITIAL_EPS
    all_episode_reward = [] # record total reward in each episode.
    for e in range(MAX_TRAIN_EPISODE):
        game.new_episode()
        t = 0 # reset the time step at the beginning of a game.
        while not game.is_episode_finished():
            current_state = game.get_state()
            current_frame = current_state.screen_buffer # return a numpy array 3xHxW
            # preprocess
            current_frame = preprocess(current_frame.transpose(1,2,0), (h, w)) # return a 30x45x3 numpy array
            # accumulate reward
            #episode_total_reward += current_reward
            # stack last 4 frames together
            if t == 0:
                # return a 30x45x3 numpy array
                current_store_state = current_frame#np.concatenate([current_frame, current_frame, current_frame, current_frame], axis=2)
            else:
                current_store_state = current_frame#np.concatenate([current_store_state[:,:,-3:], current_frame], axis=2) # 100x160x4
            
            # feed current state into network and get the q value for each action.
            current_q = session.run(q_values, feed_dict={states_ph: np.expand_dims(current_store_state, axis=0)}) # return 1x3
            current_q = np.squeeze(current_q)
            #print("*"*100)
            #print(current_q.shape)
            #print("*"*100)
            current_action, current_a_onehot = e_greedy_select(actions, current_q, eps)
            # make an action
            current_reward = game.make_action(current_action, FRAME_REPEAT)
            # get terminal input after making the current action
            terminal = game.is_episode_finished()
            # get transition info
            next_state = game.get_state()
            next_store_state = np.zeros_like(current_store_state)
            if not terminal:
                next_frame = next_state.screen_buffer
                next_frame = preprocess(next_frame.transpose(1,2,0), (h, w)) # return a 30x45x1 numpy array
                next_store_state = next_frame#np.concatenate([current_store_state[:,:,-3:], next_frame], axis=2)
        
            # push into replay cache
            replay_cache.append([current_store_state, current_a_onehot, current_reward, next_store_state, terminal])
            if len(replay_cache) > REPLAY_MEMORY:
                replay_cache.popleft()
              
            # if the agent has collected enough observation, then start to train the network.
            batch_loss = None
            if len(replay_cache) > OBSERVE:
                # sample a mini-batch from replay cache
                batch_data = random.sample(replay_cache, BATCH_SIZE)
                
                # get current_state, action, reward, next_state
                batch_sj = [x[0] for x in batch_data]
                batch_action = [x[1] for x in batch_data]
                batch_reward = [x[2] for x in batch_data]
                batch_sj1 = [x[3] for x in batch_data]
                batch_terminal = [x[4] for x in batch_data]

                #print(batch_sj[0].shape)
                #print(np.asarray(batch_sj1).shape)
                
                # get yj
                batch_yjs = []
                batch_qj1 = session.run(q_values, feed_dict={states_ph: batch_sj1})
                for i in range(BATCH_SIZE):
                    if batch_terminal[i]:
                        batch_yjs.append(batch_reward[i])
                    else:
                        batch_yjs.append(batch_reward[i] + GAMMA * np.max(batch_qj1[i]))
                        
                # train_step: update networks
                batch_loss, _ = session.run([loss, train_step],
                                            feed_dict={states_ph: batch_sj,actions_ph: batch_action, ys_ph: batch_yjs})
                
            # update t
            t += 1
            
            # update eps every 100 episodes
            if (e+1)%100 == 0:
                eps -= (eps-FINAL_EPS)/(MAX_TRAIN_EPISODE/100)
                eps = max(eps, FINAL_EPS)
                
        # after finishing each episode, print out its total reward.
        print("*"*100)
        print("Finish {} th episode at {} th time steps.".format(e, t))
        all_episode_reward.append(game.get_total_reward())
        print("Reward in {} th episode: {}".format(e, np.mean(all_episode_reward[-50:])))
        if batch_loss != None:
            print("Minibatch train loss in {} th episode: {}".format(e, batch_loss))
        print("Size of Replay Cache: {}".format(len(replay_cache)))
        print("*"*100)
        # save the model every 1000 episodes
        if (e+1)%1000 == 0:
            saver.save(session, 'saved_networks/' + scene_name + '/doom-dqn-' + scene_name, global_step = e+1)
    

if __name__ == "__main__":
    train()
    
    
    
    
    
