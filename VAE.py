import tensorflow as tf
import numpy as np
# import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
# from utils import *
from opsVAE import *

class LatentAttention():
    def __init__(self,batchsize,n_z):
        self.batchsize=batchsize
        self.n_z=n_z


    # # encoder
    # def recognition(self, input_images):
    #     with tf.variable_scope("recognition"):
    #         h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
    #         h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
    #         h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])
    #
    #         w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
    #         w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")
    #
    #     return w_mean, w_stddev

        # encoder

    def recognition(self, input_images):
        with tf.variable_scope("encoder"):
            _ = input_images
            num_channel = [32, 64, 128, 256, 256, 512]
            num_layer = np.ceil(np.log2(min(_.shape.as_list()[1:3]))).astype(np.int)
            for i in range(num_layer):
                ch = num_channel[i] if i < len(num_channel) else 512
                _ = conv2d(_, ch, self._is_train, info=not self._reuse,
                           norm=self._norm_type, name='conv{}'.format(i + 1))
            _ = conv2d(_, int(num_channel[i] / 4), self._is_train, k=1, s=1,
                       info=not self._reuse, norm='None', name='conv{}'.format(i + 2))

            w_mean = conv2d(_, self._num_class + 1, self._is_train, k=1, s=1, info=not self._reuse,
                            activation_fn=None, norm='None',
                            name='convMEAN{}'.format(i + 3))
            w_stddev = conv2d(_, self._num_class + 1, self._is_train, k=1, s=1, info=not self._reuse,
                              activation_fn=None, norm='None',
                              name='convSTDdev{}'.format(i + 3))

            # last_shape=_.get_shape()[3]
            # _ = tf.reshape(_, [-1,last_shape])
            # w_mean = fc(_, self._num_class + 1, self._is_train, info=not self._reuse, norm='None', name='w_mean_fc')
            # w_stddev = fc(_, self._num_class + 1, self._is_train, info=not self._reuse, norm='None', name='w_stddev_fc')

        return w_mean, w_stddev


