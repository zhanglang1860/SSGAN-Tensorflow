import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from vaeUtil import *
from ops import *

class LatentAttention():
    def __init__(self):
        self.mnist = input_data.read_data_sets("datasets/mnist/MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.batchsize = 100
        self._is_train=True
        self._reuse=False
        self._norm_type='batch'
        self._num_class=10
        self._h=28
        self._w=28
        self._c=1

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,(self._num_class+1)],0,1,dtype=tf.float32)
        z_mean=tf.reshape(z_mean, [self.batchsize,(self._num_class+1)])
        z_stddev = tf.reshape(z_stddev, [self.batchsize, (self._num_class + 1)])
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


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

    # decoder
    def generation(self, input):
        with tf.variable_scope("decoder"):
            _ = tf.reshape(input, [input.get_shape().as_list()[0], 1, 1, -1])
            _ = fc(_, 1024, self._is_train, info=not self._reuse, norm='None', name='fc')
            for i in range(int(np.ceil(np.log2(max(self._h, self._w))))):
                _ = deconv2d(_, max(self._c, int(_.get_shape().as_list()[-1] / 2)),
                             self._is_train, info=not self._reuse, norm=self._norm_type,
                             name='deconv{}'.format(i + 1))
            _ = deconv2d(_, self._c, self._is_train, k=1, s=1, info=not self._reuse,
                         activation_fn=tf.tanh, norm='None',
                         name='deconv{}'.format(i + 2))
            _ = tf.image.resize_bilinear(_, [self._h, self._w])

            h2 = tf.nn.sigmoid(_)

        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("VAEresults/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver()#max_to_keep=2
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10000):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("VAEresults/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()
