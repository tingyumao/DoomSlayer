import random
import vizdoom as vd
from vizdoom import *
import numpy as np
import scipy.ndimage as Simg
from time import sleep
import map_parser

from basic_ennemy_pos import basic_ennemy_x
from network import tf, DRQN
from video import VideoWriter
from memory import ReplayMemory
# from config import (
#     N_ACTIONS, LEARNING_RATE, MIN_MEM_SIZE, MAX_MEM_SIZE,
#     MAX_CPUS, TRAINING_STEPS, BATCH_SIZE, SEQUENCE_LENGTH,
#     QLEARNING_STEPS, MAX_EPISODE_LENGTH, DEATH_PENALTY,
#     KILL_REWARD, PICKUP_REWARD, GREEDY_STEPS, IGNORE_UP_TO,
#     BACKPROP_STEPS, USE_GAME_FEATURES, LEARN_Q, USE_RECURRENCE,
# )
from config import *
from ennemies import *

# Config variables
# N_ACTIONS = 7

# N_FEATURES = 1

# Need to be imported andaga created after wrap_play_random_episode
from multiprocessing import Pool, cpu_count
N_CORES = min(cpu_count(), MAX_CPUS)
N_CORES = 0
if N_CORES > 1:
    workers = Pool(N_CORES)

# Neural nets and tools
print('Building main DRQN')
main = DRQN(im_h, im_w, N_FEATURES, N_ACTIONS, 'main', LEARNING_RATE,
            use_game_features=USE_GAME_FEATURES, learn_q=LEARN_Q,
            recurrent=USE_RECURRENCE, loss_rate = LOSS_RATE)
print('Building target DRQN')
target = DRQN(im_h, im_w, N_FEATURES, N_ACTIONS, 'target', LEARNING_RATE, True,
        recurrent=USE_RECURRENCE, loss_rate = LOSS_RATE)
saver = tf.train.Saver()
mem = ReplayMemory(MIN_MEM_SIZE, MAX_MEM_SIZE)

def csv_output(*columns):
    def wrapper(func):
        def inner(*args, **kwargs):
            print("---------")
            print("::", func.__name__, "::")
            print(",".join(columns))
            return func(*args, **kwargs)
        return inner
    return wrapper



def build():
    # Neural nets and tools
    print('Building main DRQN')
    main = DRQN(im_h, im_w, N_FEATURES, N_ACTIONS, 'main', LEARNING_RATE,
                use_game_features=USE_GAME_FEATURES, learn_q=LEARN_Q,
                recurrent=USE_RECURRENCE)
    print('Building target DRQN')
    target = DRQN(im_h, im_w, N_FEATURES, N_ACTIONS, 'target', LEARNING_RATE, True,
            recurrent=USE_RECURRENCE)
    saver = tf.train.Saver()
    mem = ReplayMemory(MIN_MEM_SIZE, MAX_MEM_SIZE)

    return main, target, saver, mem



def create_game(visible = False):
    game = vd.DoomGame()
    game.load_config("scenarios/" + GAME_NAME + ".cfg")
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)

    # Ennemy detection
    walls = map_parser.parse("maps/" + GAME_NAME + ".txt")
    game.clear_available_game_variables()
    game.add_available_game_variable(vd.GameVariable.POSITION_X)  # 0
    game.add_available_game_variable(vd.GameVariable.POSITION_Y)  # 1
    game.add_available_game_variable(vd.GameVariable.POSITION_Z)  # 2

    game.add_available_game_variable(vd.GameVariable.KILLCOUNT)   # 3
    game.add_available_game_variable(vd.GameVariable.DEATHCOUNT)  # 4
    game.add_available_game_variable(vd.GameVariable.ITEMCOUNT)   # 5

    game.set_labels_buffer_enabled(True)
    game.set_window_visible(visible)

    game.init()
    return game, walls


@csv_output("mem_size", "n_games")
def bootstrap_phase(sess):
    while not mem.initialized:
        for episode in multiplay():
            if len(episode) > SEQUENCE_LENGTH:
                mem.add(episode)
        print("{},{}".format(len(mem), len(mem.episodes)))


def multiplay():
    if N_CORES <= 1:
        return [wrap_play_random_episode()]
    else:
        return workers.map(wrap_play_random_episode, range(N_CORES))

def wrap_play_random_episode(i=0):
    try:
        game, walls = create_game()
        res = play_random_episode(game, walls, skip=4)
        game.close()
        return res
    except vd.vizdoom.ViZDoomErrorException:
        print("ViZDoom ERROR")
        return []

def play_random_episode(game, walls, verbose=False, skip=1):
    game.new_episode()
    dump = []
    zoomed = np.zeros((MAX_EPISODE_LENGTH, im_h, im_w, 3), dtype=np.uint8)
    action = ACTION_SET[0]
    while not game.is_episode_finished():
        # Get screen buf
        state = game.get_state()
        S = state.screen_buffer  # NOQA

        # Resample to our network size
        h, w = S.shape[:2]
        new_image = S/255.0
        Simg.zoom(new_image, [1. * im_h / h, 1. * im_w / w, 1],
                  output=zoomed[len(dump)], order=0)
        S = zoomed[len(dump)]  # NOQA

        # Get game features an action
        # game_features = [basic_ennemy_x(state)]
        game_features = get_game_feature(state, walls)
        action = random.choice(ACTION_SET)
        reward = game.make_action(action, skip)
        dump.append((S, action, reward, game_features))
    return dump



def init_phase(sess):
    """
    Attempt to restore a model, or initialize all variables.
    Then fills replay memory with random-action games
    """
    try:
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print("Successfully loaded model")
    except:
        import traceback
        traceback.print_exc()
        init = tf.global_variables_initializer()
        sess.run(init)
        print("=== Recreate new model ! ===")



def update_target(sess):
    """Transfer learned parameters from main to target NN"""
    v = tf.trainable_variables()
    main_vars = filter(lambda x: x.name.startswith('main'), v)
    target_vars = filter(lambda x: x.name.startswith('target'), v)
    for t, m in zip(target_vars, main_vars):
        sess.run(t.assign(m.value()))


cols = ("qlearning_step", "epsilon", "reward", "steps", "loss_Q", "loss_gf")
cols += tuple("Q%d" % i for i in range(N_ACTIONS))
@csv_output(*cols)
def learning_phase(sess):
    """Reinforcement learning for Qvalues"""
    game, walls = create_game()
    max_tot_reward = {'idx':0, 'val':0}

    # From now on, we don't use game features, but we provide an empty
    # numpy array so that the ReplayMemory is still zippable
    for i in range(QLEARNING_STEPS):
        screenbuf = np.zeros((MAX_EPISODE_LENGTH, im_h, im_w, 3), dtype=np.uint8)

        # Linearly decreasing epsilon
        epsilon = max(0.1, 1 - (0.9 * i / GREEDY_STEPS))
        episode = []

        try:
            game.new_episode()
            # Initialize new hidden state
            s = 0
            h_size = 0 if not USE_RECURRENCE else main.h_size
            hidden_state = (np.zeros((1, h_size)), np.zeros((1, h_size)))
            while not game.is_episode_finished():
                # Get and resize screen buffer
                state = game.get_state()
                h, w, d = state.screen_buffer.shape
                new_image = state.screen_buffer/255.0
                Simg.zoom(new_image,
                          [1. * im_h / h, 1. * im_w / w, 1],
                          output=screenbuf[s], order=0)

                # Choose action with e-greedy network
                action_no, hidden_state = main.choose(sess, epsilon, screenbuf[s],
                                        dropout_p=DROP_OUT, state_in=hidden_state)

                action = ACTION_SET[action_no]
                reward = game.make_action(action, 4)
                # game_features = [basic_ennemy_x(state)]
                game_features = get_game_feature(state, walls)
                episode.append((screenbuf[s], action, reward, game_features))
                s += 1
            # episode = reward_reshape(episode)
            if len(episode) > SEQUENCE_LENGTH:
                mem.add(episode)
            # deaths = 1 if len(episode) != MAX_EPISODE_LENGTH else 0
            tot_reward = sum(r for (s, a, r, f) in episode)
        except vd.vizdoom.ViZDoomErrorException:
            print("ViZDoom ERROR !")
            game, walls = create_game()

        # if i % 200 == 0:
        #     make_video(sess, "videos/learning%05d.avi" % i, 3)

        if tot_reward > max_tot_reward['val'] or tot_reward == max_tot_reward['val'] and i > max_tot_reward['idx']:
            max_tot_reward['val'] = tot_reward
            max_tot_reward['idx'] = i
            saver.save(sess, "./model"+"-"+str(i)+"-"+str(int(tot_reward))+".ckpt")

        # Adapt target every 10 runs
        if i % UPDATE_GAP == 0:
            update_target(sess)

        loss_q_mean = 0

        # Then replay a few sequences
        for j in range(BACKPROP_STEPS):
            # Sample a batch and ingest into the NN
            samples = mem.sample(BATCH_SIZE, SEQUENCE_LENGTH+1)
            # screens, actions, rewards, game_features
            S, A, R, F = map(np.array, zip(*samples))

            target_q = sess.run(target.max_Q, feed_dict={
                target.batch_size: BATCH_SIZE,
                target.sequence_length: SEQUENCE_LENGTH,
                target.images: S[:, 1:],
                target.dropout_p: 1,
            })

            _, loss_q, loss_gf, qs = sess.run([main.train_step, main.q_loss, main.features_loss,main.Q], feed_dict={
                main.batch_size: BATCH_SIZE,
                main.sequence_length: SEQUENCE_LENGTH,
                main.ignore_up_to: IGNORE_UP_TO,
                main.images: S[:, :-1],
                main.target_q: target_q,
                main.gamma: GAMMA,
                main.rewards: R[:, :-1],
                main.actions: A[:, :-1],
                main.dropout_p: DROP_OUT,
                main.game_features_in: F[:, :-1]
            })

            loss_q_mean += loss_q
        loss_q_mean /= BACKPROP_STEPS
        qs = np.mean(np.mean(qs,axis =1),axis=0)
        print("{}-{}-{}-{}-{}-{}-{}".format(i, epsilon, tot_reward, len(episode), loss_q_mean, loss_gf, ",".join(map(str,qs))))

        # # Save the model periodically
        # if i > 0 and i % 200 == 0:
        #     saver.save(sess, "./model"+str(i)+".ckpt")

    game.close()



@csv_output("actual_ennemy_pos", "predicted_pos")
def testing_phase(sess):
    """Reinforcement learning for Qvalues"""
    game, walls = create_game(visible = True)


    # From now on, we don't use game features, but we provide an empty
    # numpy array so that the ReplayMemory is still zippable
    for i in range(QLEARNING_STEPS):
        screenbuf = np.zeros((im_h, im_w, 3), dtype=np.uint8)
        epsilon = 0

        try:
            # Initialize new hidden state
            total_reward = 0
            game.new_episode()
            h_size = 0 if not USE_RECURRENCE else main.h_size
            hidden_state = (np.zeros((1, h_size)), np.zeros((1, h_size)))
            while not game.is_episode_finished():
                # Get and resize screen buffer
                state = game.get_state()
                h, w, d = state.screen_buffer.shape
                new_image =state.screen_buffer/255.0
                Simg.zoom(new_image,
                          [1. * im_h / h, 1. * im_w / w, 1],
                          output=screenbuf, order=0)

                # features = sess.run(main.game_features, feed_dict={
                #     main.sequence_length: 1,
                #     main.batch_size: 1,
                #     main.images: [[screenbuf]],
                #     main.dropout_p: 1,  # No dropout in testing
                # })

                # observed_game_features = basic_ennemy_x(state)
                # predicted_game_features = features[0][0][0]
                # print("{},{}".format(observed_game_features, predicted_game_features))

                # Choose action with e-greedy network
                action_no, hidden_state = main.choose(sess, epsilon, screenbuf,
                        dropout_p=1, state_in=hidden_state)
                action = ACTION_SET[action_no]
                total_reward += game.make_action(action, 4)
                sleep(0.1)
            print(total_reward)
        except vd.vizdoom.ViZDoomErrorException:
            print("VizDoom ERROR !")
            game, walls = create_game()

    game.close()



def reward_reshape(dump):
    is_dead = len(dump) < MAX_EPISODE_LENGTH
    reward = [frame[2] for frame in dump]
    kills = [frame[4] for frame in dump]
    items = [frame[5] for frame in dump]
    kill_diff = [0] + [(kills[i] - kills[i - 1]) * KILL_REWARD for i in range(1, len(kills))]
    item_diff = [0] + [(items[i] - items[i - 1]) * PICKUP_REWARD for i in range(1, len(items))]

    reshaped_reward = [r + k + i for r, k, i in zip(reward, kill_diff, item_diff)]

    if is_dead:
        reshaped_reward[-1] -= DEATH_PENALTY

    return [
        (buffer, action, r_reward, game_features)
        for (buffer, action, _, game_features, _, _), r_reward
        in zip(dump, reshaped_reward)
    ]


def make_video(sess, filename, n_games=3):
    """Reinforcement learning for Qvalues"""
    game, walls = create_game()
    w, h = game.get_screen_width(), game.get_screen_height()
    video = VideoWriter(w, h, 25, filename)
    # sep_frame = np.zeros((w, h, 3), dtype=np.uint8)
    sep_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # From now on, we don't use game features, but we provide an empty
    # numpy array so that the ReplayMemory is still zippable
    for i in range(n_games):
        screenbuf = np.zeros((im_h, im_w, 3), dtype=np.uint8)
        epsilon = 0

        try:
            # Initialize new hidden state
            total_reward = 0
            game.new_episode()
            h_size = main.h_size if USE_RECURRENCE else 0
            hidden_state = (np.zeros((1, h_size)), np.zeros((1, h_size)))
            while not game.is_episode_finished():
                # Get and resize screen buffer
                state = game.get_state()
                for i in range(3):
                    video.add_frame(state.screen_buffer)
                h, w, d = state.screen_buffer.shape
                new_image =state.screen_buffer/255.0
                Simg.zoom(new_image,
                          [1. * im_h / h, 1. * im_w / w, 1],
                          output=screenbuf, order=0)

                # Choose action with e-greedy network
                action_no, hidden_state = main.choose(sess, epsilon, screenbuf,
                        dropout_p=1, state_in=hidden_state)
                action = ACTION_SET[action_no]
                total_reward += game.make_action(action, 4)
        except vd.vizdoom.ViZDoomErrorException:
            print("VizDoom ERROR !")
            game, walls = create_game()

        for i in range(25):
            video.add_frame(sep_frame)
    video.close()
    game.close()
