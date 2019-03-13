from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import huber_loss
from util import log
from encoder import Encoder
from decoder import Decoder


class VAEModel(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.h = self.config.h
        self.w = self.config.w
        self.c = self.config.c

        self.n_z = config.n_z
        self.norm_type = config.norm_type
        self.num_class = self.config.num_class

        self.image = tf.placeholder(tf.float32, [None, 784])
        self.image_matrix = tf.reshape(self.image, [-1, 28, 28, 1])

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')
        self.buildVAEmodel(is_train=is_train)


    def buildVAEmodel(self, is_train=True):
       # build loss and accuracy {{{
        def build_loss(decoded_flat, z_mean, z_stddev):
            generation_loss = -tf.reduce_sum(
                self.image * tf.log(1e-8 + decoded_flat) + (1 - self.image) * tf.log(1e-8 + 1 - decoded_flat), 1)

            latent_loss = 0.5 * tf.reduce_sum(
                tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)
            cost = tf.reduce_mean(generation_loss + latent_loss)
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
            return generation_loss, latent_loss, cost
        # }}}

        # encoder {{{
        # =========
        E = Encoder('Encoder', self.norm_type, is_train, self.batch_size,self.n_z, self.num_class)
        w_mean, w_stddev = E(self.image_matrix)

        w_mean = tf.reshape(w_mean, [self.batch_size, self.n_z])
        w_stddev = tf.reshape(w_stddev, [self.batch_size, self.n_z])
        samples = tf.random_normal([self.batch_size, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = w_mean + (w_stddev * samples)
        # }}}

        # encoder {{{
        # =========
        De = Decoder('Decoder', self.norm_type, is_train, self.h, self.w, self.c)
        self.decoder_images = De(guessed_z)
        # }}}

        decoded_flat = tf.reshape(self.decoder_images, [self.batch_size, 28 * 28])






        self.reconstruction_loss, self.latent_loss, self.vae_total_cost = \
            build_loss(decoded_flat, w_mean, w_stddev)

        tf.summary.scalar("vae/loss/vae_loss", self.vae_total_cost)
        tf.summary.scalar("vae/loss/reconstruction_loss", self.reconstruction_loss)
        tf.summary.scalar("vae/loss/latent_loss", self.latent_loss)

        log.warn('\033[93mSuccessfully loaded the VAE model.\033[0m')
