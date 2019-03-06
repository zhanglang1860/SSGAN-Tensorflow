import tensorflow as tf
import numpy as np

x = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]]

a = tf.transpose(x, [0, 1, 2])
b = tf.transpose(x, [0, 2, 1])
c = tf.transpose(x, [1, 0, 2])
d = tf.transpose(x, [1, 2, 0])
e = tf.transpose(x, [2, 1, 0])
f = tf.transpose(x, [2, 0, 1])
fshape = f.get_shape().as_list()
g = tf.reshape(f, [-1])
shape_size=fshape[1]*fshape[2]
covariance_matrix_shape = np.dtype('int32').type(shape_size)
h = tf.reshape(g, [-1, covariance_matrix_shape])

with tf.Session() as sess:
    # print ('---------------')
    # print (sess.run(a))
    # print ('---------------')
    # print (sess.run(b))
    # print ('---------------')
    # print (sess.run(c))
    # print ('---------------')
    # print (sess.run(d))
    # print ('---------------')
    # print (sess.run(e))
    print (x)
    print ('-----covariance_matrix_shape----------')
    print (covariance_matrix_shape)
    print ('---------fffff------')
    print (sess.run(f))
    print ('---------gggggg------')
    print (sess.run(g))
    print ('----------hhhhhhh-----')
    print (sess.run(h))
    print ('---------------')
