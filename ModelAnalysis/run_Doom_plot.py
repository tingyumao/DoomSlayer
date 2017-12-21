"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
from vizdoom import Mode

from RL_brain_doom_plot import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import preprocess


ACTIONS_NUM = 8
INITIAL_EPS = 1.0
FINAL_EPS = 0.1
GAMMA = 0.9
LAST_FRAME_NUM = 1
FRAME_REPEAT = 8
OBSERVE = 3000
REPLAY_MEMORY = 10000
MAX_TRAIN_EPISODE = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 64


cfg_path = "./defend_the_line.cfg"
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

actions = set_actions()

#env = gym.make('MountainCar-v0')
#env = env.unwrapped
#env.seed(21)
MEMORY_SIZE = 10000

sess = tf.Session()
#with tf.variable_scope('natural_DQN'):
#    RL_natural = DQNPrioritizedReplay(
#        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
#        e_greedy_increment=0.00005, sess=sess, prioritized=False,
#    )

img_w, img_h = 120, 90

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=8, width=img_w, height=img_h, n_features=1, memory_size=MEMORY_SIZE, batch_size=64,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True)
sess.run(tf.global_variables_initializer())

# tf.glorot_normal_initializer
#sess.run(tf.glorot_normal_initializer()(tf.global_variables()))

def train(RL):
	total_steps = 0
	steps = []
	episodes = []
	plot_reward = np.zeros(2000)
    #for i_episode in range(2000):
	for i_episode in range(2000):
        #observation = env.reset()
        # Starts a new episode
		game.new_episode()
		episode_reward = 0
		while True:
            # env.render()
			raw_state = game.get_state()
			observation = preprocess(raw_state.screen_buffer.transpose(1, 2, 0), (img_h, img_w))
			action = RL.choose_action(observation)
			reward = game.make_action(actions[action], FRAME_REPEAT)

			episode_reward += reward

			if game.is_episode_finished():
				observation_ = np.zeros_like(observation)
			else:    
				raw_state = game.get_state()
				observation_ = preprocess(raw_state.screen_buffer.transpose(1, 2, 0), (img_h, img_w))
            #observation_, reward, done, info = env.step(action)

            #if done: reward = 10

			RL.store_transition(observation, action, reward, observation_)

			if total_steps > MEMORY_SIZE:
				RL.learn()

			if game.is_episode_finished():
				print('episode ', i_episode, ' finished. ', "reward: ", episode_reward)
				steps.append(total_steps)
				episodes.append(i_episode)
				break

			observation = observation_
			total_steps += 1
		plot_reward[i_episode] = episode_reward
	return np.vstack((episodes, steps)), plot_reward

#his_natural = train(RL_natural)
his_prio, plot_reward = train(RL_prio)
np.savetxt("DQNP_reward_plot.csv", plot_reward, delimiter = ",")
loss_list = RL_prio.learn()
plot_loss = np.zeros(len(loss_list))
inx = 0
for item in loss_list:
	plot_loss[inx] = item
	inx = inx +1
np.savetxt("DQNP_loss_plot.csv",plot_loss, delimiter = ",")
# import csv
# with open('DQNP_loss_plot.csv', 'wb') as myfile:
# 	wr = csv.writer(myfile,delimiter=',')
# 	wr.writerow(plot_loss)

# compare based on first success
#plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

game.close()


