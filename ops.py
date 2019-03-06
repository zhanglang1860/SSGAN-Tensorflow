import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
from util import log
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tflearn.layers.conv import grouped_conv_2d
import tensorflow_extentions as tfe
import tensornet
import sys


sys.path.append('../../../')


def print_info(name, shape, activation_fn):
    log.info('{}{} {}'.format(
        name,  '' if activation_fn is None else ' ('+activation_fn.__name__+')',
        shape))

def split_dimension_to_rank_multiple(x,d):
    if d==4:
        dimensionTemp = [4, 4, 4, 4]
        out = np.array([4, 4, 4, 4])
        if x%64==0:
            out[3]=x/64
        else:
            if x%16==0:
                if x%32==0:
                    out[3] = x/32
                    out[2] = 2
                else:
                    out[2]=1
                    out[3]=x/16
            else:
                out[1]=x/4
                out[2]=1
                out[3]=1
    else:
        dimensionTemp = [4, 4, 4]
        out = np.array([4, 4, 4])
        if x % 16 == 0:
            out[2] = x / 16
        else:
            if x % 4 == 0:
                if x % 4 == 0:
                    out[3] = x / 8
                    out[2] = 2
                else:
                    out[2] = 1
                    out[3] = x / 4
            else:
                out[1] = 1
                out[2] = 1
                out[0] = x
    out.astype(np.int32)
    return out



def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def llrelu(x, leak=0.1, name="llrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def instance_norm(input):
    """
    Instance normalization
    """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def norm_and_act(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    """
    Apply normalization and/or activation function
    """
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True,
                    updates_collections=None,
                )
            elif norm == 'instance':
                _ = instance_norm(_, is_train)
            elif norm == 'None':
                _ = _
            else:
                raise NotImplementedError
    return _


def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [1])
    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=epsilon)


def _covariance(x, diag):
    """Defines the covariance operation of a matrix.

    Args:
    x: a matrix Tensor. Dimension 0 should contain the number of examples.
    diag: if True, it computes the diagonal covariance.

    Returns:
    A Tensor representing the covariance of x. In the case of
    diagonal matrix just the diagonal is returned.
    """

    f = tf.transpose(x, [2, 0, 1])
    fshape = f.get_shape().as_list()
    g = tf.reshape(f, [-1])
    shape_size = fshape[1] * fshape[2]
    covariance_matrix_shape = np.dtype('int32').type(shape_size)
    h = tf.reshape(g, [-1, covariance_matrix_shape])
    h = tf.transpose(h, [1, 0])

    num_points = math_ops.to_float(array_ops.shape(h)[0])
    h -= math_ops.reduce_mean(h, 0, keepdims=True)
    if diag:
        cov = math_ops.reduce_sum(math_ops.square(h), 0, keepdims=True) / (num_points - 1)
    else:
        cov = math_ops.matmul(h, h, transpose_a=True) / (num_points - 1)

    batch_norm(cov)
    return cov


def Global_Covariance_Matrix(x, diag):
    x = norm_and_act(x, is_train=True, norm='batch', activation_fn=lrelu)
    covariance_matrix_shape = x.get_shape().as_list()
    for i in range(0, covariance_matrix_shape[0]):
        each_image_covariance_matrix = _covariance(x[i], diag)
        each_image_covariance_matrix = tf.reshape(each_image_covariance_matrix,
                                                  [1, covariance_matrix_shape[3], covariance_matrix_shape[3]])
        if i == 0:
            result = each_image_covariance_matrix
        else:
            result = tf.concat([result, each_image_covariance_matrix], axis=0)
    return result


def excitation_layer(input_x,out_dim,orignialInput,is_train=True,name="GsoPexcitation"):
    with tf.variable_scope(name):
        excitation = norm_and_act(input_x, is_train, norm='batch', activation_fn=lrelu)
        excitation = slim.conv2d(excitation, out_dim, [1, 1], stride=1, activation_fn=None)
        excitation = tf.sigmoid(excitation)
        # excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = orignialInput * excitation
    return scale


def squeeze_excitation_layer(input_x, is_train=True, name="GsoP"):
    with tf.variable_scope(name):
        orignialInput = input_x
        out_dim = input_x.get_shape().as_list()[3]
        covariance_matrix_shape = out_dim / 8
        covariance_matrix_shape = np.dtype('int32').type(covariance_matrix_shape)
        squeeze = slim.conv2d(input_x, covariance_matrix_shape, [1, 1], stride=1, activation_fn=None)

        squeeze = Global_Covariance_Matrix(squeeze, False)

        squeeze = tf.reshape(squeeze, [-1, 1, covariance_matrix_shape, covariance_matrix_shape])

        # squeeze = norm_and_act(squeeze, is_train, norm='batch', activation_fn=None)




        # excitation = conv2d_group(squeeze, covariance_matrix_shape, name)
        excitation = grouped_conv_2d(squeeze, 4, [1, covariance_matrix_shape],  strides=1, padding='VALID', name=name)

        scale = excitation_layer(excitation, out_dim, orignialInput, name=name)
    return scale



def conv2d_group(squeeze, covariance_matrix_shape, name="GsoPgroupCov"):
    with tf.variable_scope(name):
        excitation = tfe.grouped_convolution(squeeze, 4 * covariance_matrix_shape, [1, covariance_matrix_shape],
                                             covariance_matrix_shape,
                                             padding='VALID',
                                             activation_fn=None,
                                             name=name)
    return excitation



def conv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
           activation_fn=lrelu, norm='batch', name="conv2d"):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k, k], stride=s, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def conv2dtensorNet(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
           activation_fn=lrelu, norm='batch', name="conv2dTensorNet"):
    with tf.variable_scope(name):
        out_mode = split_dimension_to_rank_multiple(output_shape, 3)
        inp_shape = input.get_shape().as_list()[3]
        input_mode = split_dimension_to_rank_multiple(inp_shape, 3)
        _ = tensornet.layers.tt_conv_full(input, [k, k], input_mode, out_mode,
                                          np.array([16, 16, 16, 1], dtype=np.int32),
                                          [s, s], biases_initializer=None, scope=name)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _



def deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
             activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def bilinear_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
                      activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2dtensorNet(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def nn_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
                activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_nearest_neighbor(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def fc(input, output_shape, is_train, info=False, norm='batch',
       activation_fn=lrelu, name="fc"):
    with tf.variable_scope(name):
        _ = slim.fully_connected(input, output_shape, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def fcTensorNet(input, output_shape, is_train, info=False, norm='batch',
       activation_fn=lrelu, name="fcTensorNet"):
    with tf.variable_scope(name):
        inp_shape = input.get_shape().as_list()[0]
        inp = tf.reshape(input, [inp_shape, -1])
        inp_shape_split = inp.get_shape().as_list()[1]
        input_mode=split_dimension_to_rank_multiple(inp_shape_split,4)
        out_mode = split_dimension_to_rank_multiple(output_shape, 4)
        _ = tensornet.layers.tt(inp, input_mode, out_mode,  np.array([1, 40, 40, 40, 1], dtype=np.int32),
                                biases_initializer=None, scope=name)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
        result=tf.reshape(_, [inp_shape, 1, 1, -1])
    return result
