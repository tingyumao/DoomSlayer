# coding: utf-8
a_size = 3
ACTION_DIM = 2 ** a_size - 2  # remove (True, True, True) and (True, True, False)[Left, Right, Forward]
model_path = './model/'
model_file = 'model-1750.cptk'

IS_TRAIN = True