import numpy as np
# Replay memory minimum and maximum size
MIN_MEM_SIZE, MAX_MEM_SIZE = 2000, 80000
# Batch size for NN ingestion
BATCH_SIZE = 32
# Sequence length for NN ingestion
SEQUENCE_LENGTH = 8
# Number of states to ignore when computing loss
IGNORE_UP_TO = 4
# Maximum episode duration, in frames
# MAX_EPISODE_LENGTH = 125  # 500 with a frame skip of 4 #basic
# MAX_EPISODE_LENGTH = 525 # dealy co
# Number of training steps
# TRAINING_STEPS = 1000
# Number of backpropagation steps to execute after each episode
BACKPROP_STEPS = 15
# Number of training steps
QLEARNING_STEPS = 6000
# Number of steps during which epsilon should be decreased
GREEDY_STEPS = 4000
# Maximum number of cores to use
MAX_CPUS = 32
# Learning rate for tensorflow optimizers
LEARNING_RATE = 0.001
# Use the game features in the learning phase
USE_GAME_FEATURES = True
# Use LSTM or simple DQN
USE_RECURRENCE = True
# Learn Q in the learning phase
LEARN_Q = True

im_w, im_h = 160, 128

LOSS_RATE = 0.1

GAMMA = 0.99
DROP_OUT = 0.75

UPDATE_GAP = 50

GAME_NAME = 'defend_the_line'
N_ACTIONS = 3
MAX_EPISODE_LENGTH = 2000 # defend_the_line

N_FEATURES = 6
ACTION_SET = np.eye(N_ACTIONS, dtype=np.uint32).tolist()
SECTION_SEPARATOR = "------------"


DEATH_PENALTY = 25
KILL_REWARD = 100
PICKUP_REWARD = 4

