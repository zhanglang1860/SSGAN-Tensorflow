import numpy as np
import tensorflow as tf
from ops import fcTensorNet


class Generator(object):
    def __init__(self, name, h, w, c, norm_type, deconv_type, is_train,split_dimension_core,tt_rank):
        self.name = name
        self._h = h
        self._w = w
        self._c = c
        self._norm_type = norm_type
        self._deconv_type = deconv_type
        self._is_train = is_train
        self._reuse = False
        self.split_dimension_core = split_dimension_core
        self.tt_rank = tt_rank

    def __call__(self, input):
        if self._deconv_type == 'bilinear':
            from ops import bilinear_deconv2d as deconv2d
        elif self._deconv_type == 'nn':
            from ops import nn_deconv2d as deconv2d
        elif self._deconv_type == 'transpose':
            from ops import deconv2d
        else:
            raise NotImplementedError
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = tf.reshape(input, [input.get_shape().as_list()[0], 1, 1, -1])
            _ = fcTensorNet(_, 8192, self._is_train, info=not self._reuse, norm='None', name='fc', split_dimension_core=self.split_dimension_core,tt_rank=self.tt_rank)
            # for i in range(int(np.ceil(np.log2(max(self._h, self._w))))):
            for i in range(7):
                _ = deconv2d(_, max(self._c, int(_.get_shape().as_list()[-1] / 2)),
                             self._is_train, info=not self._reuse, norm=self._norm_type,
                             name='deconv{}'.format(i + 1),split_dimension_core=self.split_dimension_core,tt_rank=self.tt_rank)

            # for index in range(int(np.ceil(np.log2(int(_.get_shape().as_list()[-1]))))):
            for index in range(6):
                _ = deconv2d(_, self._c, self._is_train, k=1, s=1, info=not self._reuse,
                             norm=self._norm_type,
                             name='deconv{}'.format(i + index + 2),split_dimension_core=self.split_dimension_core,tt_rank=self.tt_rank)

            _ = deconv2d(_, self._c, self._is_train, k=1, s=1, info=not self._reuse,
                         activation_fn=tf.tanh, norm='None',
                         name='deconv{}'.format(i + index + 3),split_dimension_core=self.split_dimension_core,tt_rank=self.tt_rank)
            _ = tf.image.resize_bilinear(_, [self._h, self._w])

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _
