# encoding: UTF-8

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import ticker
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


class Evaluator(object):
    def __init__(self):
        cwd = os.path.dirname(__file__)
        tf.reset_default_graph()
        imported_meta = tf.train.import_meta_graph(cwd + "/model_final.meta")

        self.sess = tf.Session()
        imported_meta.restore(self.sess, tf.train.latest_checkpoint(cwd + '/'))

    def evaluate(self, test_dict):
        images_2d = [value[0].reshape(28, 28, 1) if value[0] is not None else np.zeros((28, 28, 1)) for value in
                     test_dict.values()]
        labels = [value[1] for value in
                     test_dict.values()]
        # images_ndarray = np.empty((len(images_2d), 28, 28, 1))
        images_ndarray = np.array(images_2d)
        #         cv2.waitKey()
        #         cv2.destroyAllWindows()
        # print(result_dict.values())
        Y = self.sess.graph.get_tensor_by_name('Y:0')

        mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

        # input_data = {'X:0': mnist.test.images, 'tst:0': True, 'pkeep:0': 1, 'pkeep_conv:0': 1.0}
        input_data = {'X:0': images_2d, 'tst:0': True, 'pkeep:0': 1, 'pkeep_conv:0': 1.0}

        with self.sess as sess:
            result = sess.run(Y,input_data)
            predictions = np.argmax(result, 1)
            # labels = np.argmax(mnist.test.labels, 1)
            print(predictions)
            for i in range(len(predictions)):
                if predictions[i] != labels[i]:
                    plt.figure(1)
                    plt.subplot(211)
                    plt.axis("off")
                    plt.imshow(input_data['X:0'][i].reshape(28,28), cmap='gray')
                    ax = plt.subplot(212)
                    x_pos = np.arange(10)
                    plt.bar(x_pos, result[i])
                    ax.xaxis.set_ticks(np.arange(0, 10, 1))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1d'))
                    plt.show()
                    # cv2.namedWindow(str(predictions[i]), cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow(str(predictions[i]), 600, 600)
                    # cv2.imshow(str(predictions[i]), mnist.test.images[i])
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
            return result
