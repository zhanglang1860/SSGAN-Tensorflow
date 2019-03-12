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


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev


