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

from RL_brain_ddoom import DDQNPrioritizedReplay
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

game.set_window_visible(False)

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

with tf.variable_scope('DDQN_with_prioritized_replay'):
    RL_prio = DDQNPrioritizedReplay(
        n_actions=8, width=img_w, height=img_h, n_features=1, memory_size=MEMORY_SIZE, batch_size=64,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# load model
checkpoint = tf.train.get_checkpoint_state("./DDoom")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Failed to load model. Please check if the old model exists")
    #return


# tf.glorot_normal_initializer
#sess.run(tf.glorot_normal_initializer()(tf.global_variables()))

def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        #observation = env.reset()
        # Starts a new episode
        game.new_episode()
        episode_reward = 0
        while True:
            # env.render()
            raw_state = game.get_state()
            img = preprocess(raw_state.screen_buffer.transpose(1, 2, 0), (img_h, img_w))
            img = np.expand_dims(img, axis=0)
            angle = game.get_game_variable(GameVariable.ANGLE)
            health = game.get_game_variable(GameVariable.HEALTH)
            measures = np.asarray([((angle+90)%360)/360., health / 100.]).astype("float32")
            measures = measures[np.newaxis, :]
            action = RL.play([img, measures])
            reward = game.make_action(actions[action], FRAME_REPEAT)

            episode_reward += reward

            if game.is_episode_finished():
                print('episode ', i_episode, ' finished. ', "reward: ", episode_reward)
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            total_steps += 1


    return np.vstack((episodes, steps))

#his_natural = train(RL_natural)
his_prio = train(RL_prio)

# compare based on first success
#plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DDQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

game.close()


