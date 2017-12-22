#!/usr/bin/env python
import tensorflow as tf
from os import system
from optparse import OptionParser

if __name__ == '__main__':
    # system('cat config.py local_config.py')
    # system('git show | head -1')

    parser = OptionParser()
    parser.add_option("-B", "--no-bootstrap", dest="bootstrap",
                      action="store_false", default=True,
                      help="Do not populate replay memory with random actions")
    parser.add_option("-Q", "--no-learning", dest="learning",
                      action="store_false", default=True,
                      help="Do not perform the learning phase")
    parser.add_option("-R", "--no-testing", dest="testing",
                      action="store_false", default=True,
                      help="Do not perform the testing phase")
    parser.add_option("-V", "--video-record", dest="record",
                      action="store", default="video.mp4", type="string",
                      help="Records")
    options, args = parser.parse_args()

    with tf.Session() as sess:
        # from agent import (
        #     init_phase, bootstrap_phase,
        #     learning_phase, testing_phase,
        #     update_target, make_video
        # )

        from agent import *

        # main, target, saver, mem = build()

        init_phase(sess)

        if options.bootstrap:
            bootstrap_phase(sess)

        update_target(sess)

        if options.learning:
            learning_phase(sess)

        # if options.testing:
            # testing_phase(sess)

        # if options.record:
        #     make_video(sess, "videos/final.mp4", 3)
