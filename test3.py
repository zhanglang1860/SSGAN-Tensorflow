import tensorflow as tf
W = tf.constant([[-2.,12.,6.],[3.,2.,8.]], )
mean,var = tf.nn.moments(W, axes = [1])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    resultMean = sess.run(mean)
    print(resultMean)
    resultVar = sess.run(var)
    print(resultVar)
