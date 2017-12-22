#!/usr/bin/env python

from __future__ import print_function

from time import sleep
from vizdoom import * # NOQA
import ennemies
from ennemies import *
import map_parser

game = DoomGame()
game.set_labels_buffer_enabled(True)
game.set_depth_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

# game.load_config("scenarios/deathmatch.cfg")
game.load_config("scenarios/defend_the_line.cfg")
game.add_game_args("+freelook 1")

game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)

game.clear_available_game_variables()
game.add_available_game_variable(GameVariable.POSITION_X)
game.add_available_game_variable(GameVariable.POSITION_Y)
game.add_available_game_variable(GameVariable.POSITION_Z)


game.init()

episodes = 10

walls = map_parser.parse("maps/defend_the_line.txt")

for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()
    j = 0
    while not game.is_episode_finished():
        state = game.get_state()
        j += 1
        if j % 15 == 0:
            # print([x.object_name for x in ennemies.get_visible_ennemies(state, walls)])
            # print(ennemies.has_visible_entities(state, walls))
            # print(ennemies.get_game_feature(state,walls))
            print(ennemies.min_relative_pos(state,ENNEMIES))

        game.advance_action()

    sleep(0.5)

game.close()
