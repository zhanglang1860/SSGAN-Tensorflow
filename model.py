from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import huber_loss
from util import log
from generator import Generator
from discriminator import Discriminator
import numpy as np

class Model(object):

    def __init__(self, config, growth_rate, depth,
                 total_blocks, keep_prob,
                  nesterov_momentum, model_type,
                 debug_information=False,
                 is_train=True,
                 reduction=1.0,
                 bc_mode=False):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.h = self.config.h
        self.w = self.config.w
        self.c = self.config.c
        self.num_class = self.config.num_class
        self.n_z = config.n_z


        # create placeholders for the input


        # tf.summary.scalar("loss/recon_wieght", self.recon_weight)

        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob

        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type

        # self.batches_step = 0

        self.images = tf.placeholder(
            name='input_images', dtype=tf.float32,
            shape=[self.batch_size, self.h, self.w, self.c,1],
        )
        self.labels = tf.placeholder(
            name='labels', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )



        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

        self.is_training_denseNet = tf.placeholder_with_default(bool(is_train), [], name='is_training')


        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, is_training=None):
        fd = {
            self.images: batch_chunk['image'],  # [bs, h, w, c]
            self.labels: batch_chunk['label'],  # [bs, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        # Weight annealing
        # if step is not None:
        #     fd[self.recon_weight] = min(max(0, (1500 - step) / 1500), 1.0)*10
        return fd






    def build(self, is_train=True):

        n = self.num_class

        # build loss and accuracy {{{
        def build_loss(prediction, logits):
            # alpha = 0.9

            # Discriminator/classifier loss
            cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels, name='cross_entropy_per_example'), name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            correct_prediction = tf.equal(
                tf.argmax(prediction, 1),
                tf.argmax(self.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return cross_entropy_mean, correct_prediction, accuracy, tf.add_n(tf.get_collection('losses'), name='total_loss')
        # }}}




        # Discriminator {{{
        # =========
        D = Discriminator('Discriminator', self.num_class,self.h, self.w, self.c,self.growth_rate, self.depth,
                 self.total_blocks, self.keep_prob,
                  self.model_type,
                 is_train=True,
                 reduction=self.reduction,
                 bc_mode=self.bc_mode)
        prediction, logits = D(self.images)
        self.all_preds = prediction
        self.all_targets = self.labels

        # }}}

        self.cross_entropy, self.correct_prediction, self.accuracy, _ = \
            build_loss(prediction, logits)

        tf.summary.scalar("accuracy", self.accuracy)
        # tf.summary.scalar("loss/correct_prediction",  self.correct_prediction)
        tf.summary.scalar("loss/cross_entropy", self.cross_entropy)
        # tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
        # tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
        # tf.summary.image("img/fake", fake_image)
        # tf.summary.image("img/real", self.image, max_outputs=1)
        # tf.summary.image("label/target_real", tf.reshape(self.label, [1, self.batch_size, n, 1]))
        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
