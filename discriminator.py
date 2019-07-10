#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from ops import conv2d
from ops import squeeze_excitation_layer
from util import log
from ops import depthwise_conv2d
from ops import fc
from ops import grouped_conv2d_Discriminator
from ops import grouped_conv2d_Discriminator_valid
from ops import grouped_conv2d_Discriminator_one



class Discriminator(object):
    def __init__(self, name, num_class, norm_type, is_train,h, w, c):
        self.name = name
        self._num_class = num_class
        self._norm_type = norm_type
        self._is_train = is_train
        self._reuse = False
        self._h = h
        self._w = w
        self._c = c

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = input

            _ = squeeze_excitation_layer(_ , is_train=True, name="DiscriminatorGsoP01")

            _ = depthwise_conv2d(_, 3, self._is_train, info=not self._reuse, norm=self._norm_type,
                                 name='DiscriminatordepthwiseConv1')

            _ = squeeze_excitation_layer(_, is_train=True, name="DiscriminatorGsoP02")




            number_filters_each_group = 3
            _ = grouped_conv2d_Discriminator(_, number_filters_each_group * self._c, self._c, self._is_train,
                                   info=not self._reuse, norm=self._norm_type, name='DiscriminatorGroupConv1')

            number_filters_each_group = 6
            _ = grouped_conv2d_Discriminator(_, number_filters_each_group * self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse, norm=self._norm_type,
                                             name='DiscriminatorGroupConv2')

            _ = squeeze_excitation_layer(_, is_train=True, name="DiscriminatorGsoP03")

            number_filters_each_group = 6
            _ = grouped_conv2d_Discriminator(_, number_filters_each_group * self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse, norm=self._norm_type,
                                             name='DiscriminatorGroupConv3')

            number_filters_each_group = 6
            _ = grouped_conv2d_Discriminator(_, number_filters_each_group * self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse,  k=1, s=1,norm=self._norm_type,
                                             name='DiscriminatorGroupConv4')
            _ = squeeze_excitation_layer(_, is_train=True, name="DiscriminatorGsoP04")

            number_filters_each_group = 4
            _ = grouped_conv2d_Discriminator(_, number_filters_each_group * self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse, k=1, s=1, norm=self._norm_type,
                                             name='DiscriminatorGroupConv5')

            number_filters_each_group = 1
            _ = grouped_conv2d_Discriminator(_, number_filters_each_group * self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse, k=1, s=1, norm=self._norm_type,
                                             name='DiscriminatorGroupConv6')

            _ = squeeze_excitation_layer(_, is_train=True, name="DiscriminatorGsoP05")

            num_channel = [32, 64, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            num_layer = np.ceil(np.log2(min(_.shape.as_list()[1:3]))).astype(np.int)

            for i in range(5):
                ch = num_channel[i] if i < len(num_channel) else 512
                _ = grouped_conv2d_Discriminator_valid(_,  self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse, norm=self._norm_type,
                                             name='conv{}'.format(i + 1))
                if i == 1:
                    _ = squeeze_excitation_layer(_, is_train=True, name='GsoP{}'.format(i + 5))

            _ = grouped_conv2d_Discriminator_one(_,  self._c,
                                             self._c, self._is_train,
                                             info=not self._reuse, norm=self._norm_type,
                                             name='conv{}'.format(i + 2))

            _ = conv2d(_, 45, self._is_train, k=1, s=1, info=not self._reuse,
                       name='conv{}'.format(i + 3))

            _ = conv2d(_, 25, self._is_train, k=1, s=1, info=not self._reuse,
                       name='conv{}'.format(i + 4))

            _ = conv2d(_, 12, self._is_train, k=1, s=1, info=not self._reuse,
                       name='conv{}'.format(i + 5))

            _ = conv2d(_, 6, self._is_train, k=1, s=1, info=not self._reuse,
                       name='conv{}'.format(i + 6))


            _ = conv2d(_, self._num_class + 1, self._is_train, k=1, s=1, info=not self._reuse,
                       activation_fn=None, norm='None',
                       name='conv{}'.format(i + 7))
            _ = tf.squeeze(_)
            if not self._reuse: 
                log.info('discriminator output {}'.format(_.shape.as_list()))
            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return tf.nn.sigmoid(_), _
