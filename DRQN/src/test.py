#!/usr/bin/env python
import tensorflow as tf
from os import system
from optparse import OptionParser


if __name__ == '__main__':
    with tf.Session() as sess:
        # from agent import (
        #     init_phase, bootstrap_phase,
        #     learning_phase, testing_phase,
        #     update_target, make_video
        # )

        from agent import *

        try:
            saver = tf.train.import_meta_graph('model5000.ckpt.meta')
            saver.restore(sess, './model5200.ckpt')
            print("Successfully loaded model")
            print(tf.train.latest_checkpoint('./'))
        except:
            import traceback
            traceback.print_exc()
            init = tf.global_variables_initializer()
            sess.run(init)
            print("=== Recreate new model ! ===")

        update_target(sess)

        make_video(sess, "final.avi", 3)

        testing_phase(sess)



        # if options.record:
        #     make_video(sess, "videos/final.mp4", 3)
