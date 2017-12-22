#!/usr/bin/env python

"""
This is a simple test program.

The agent will randomly move right or left and shoot at the enemy. 
It will last for 10 episodes. For each episode, it will terminate 
when the agent shoot at the enemy or it reaches the maximum number 
of actions, as setting in basic.cfg(episode_timeout = 300).
"""

from vizdoom import *
import random
import time

game = DoomGame()
game.load_config("../scenarios/defend_the_line.cfg")
#game.set_mode(Mode.SPECTATOR) 
#game.set_window_visible(False)
game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

episodes = 1
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer # return a numpy array 3x240x320
        misc = state.game_variables
        angle = game.get_game_variable(GameVariable.ANGLE)
        health = game.get_game_variable(GameVariable.HEALTH)
        print("angle = {}, health = {}".format(angle, health))
        # make_action take a distribution as input? So it also do the 
        # e-greedy selection inside?
        reward = game.make_action(random.choice(actions), 10)
        print("\treward:", reward)
        time.sleep(0.1)
    print("Result:", game.get_total_reward())
    time.sleep(2)
