import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
from util import log
import sys


sys.path.append('../../../')
import tensornet

def print_info(name, shape, activation_fn):
    log.info('{}{} {}'.format(
        name,  '' if activation_fn is None else ' ('+activation_fn.__name__+')',
        shape))

def split_dimension_to_rank_multiple(inp_shape,split_dimension_core):

    if split_dimension_core==4:
        if inp_shape==1048576:
            out = np.array([32, 32, 32, 32])
        else:
            if inp_shape==524288:
                out = np.array([32, 32, 16, 32])
            else:
                if inp_shape == 262144:
                    out = np.array([32, 16, 16, 32])
                else:
                    if inp_shape == 131072:
                        out = np.array([16, 16, 16, 32])
                    else:
                        if inp_shape == 65536:
                            out = np.array([16, 16, 16, 16])
                        else:
                            if inp_shape == 32768:
                                out = np.array([16, 16, 8, 16])
                            else:
                                if inp_shape == 16384:
                                    out = np.array([16, 8, 8, 16])
                                else:
                                    if inp_shape == 8192:
                                        out = np.array([8, 8, 8, 16])
                                    else:
                                        if inp_shape == 4096:
                                            out = np.array([8, 8, 8, 8])
                                        else:
                                            if inp_shape == 2048:
                                                out = np.array([8, 4, 8, 8])
                                            else:
                                                if inp_shape == 1024:
                                                    out = np.array([8, 4, 4, 8])
                                                else:
                                                    if inp_shape == 512:
                                                        out = np.array([8, 4, 4, 4])
                                                    else:
                                                        if inp_shape == 256:
                                                            out = np.array([4, 4, 4, 4])
                                                        else:
                                                            if inp_shape == 128:
                                                                out = np.array([4, 2, 4, 4])
                                                            else:
                                                                if inp_shape == 64:
                                                                    out = np.array([4, 2, 2, 4])
                                                                else:
                                                                    if inp_shape == 32:
                                                                        out = np.array([2, 2, 2, 4])
                                                                    else:
                                                                        if inp_shape == 16:
                                                                            out = np.array([2, 2, 2, 2])
                                                                        else:
                                                                            if inp_shape == 8:
                                                                                out = np.array([2, 2, 1, 2])
                                                                            else:
                                                                                if inp_shape == 4:
                                                                                    out = np.array([2, 1, 1, 2])
    else:
        if split_dimension_core==5:
            if inp_shape == 1048576:
                out = np.array([32, 32, 32, 8,4])
            else:
                if inp_shape == 524288:
                    out = np.array([32, 32, 16, 8,4])
                else:
                    if inp_shape == 262144:
                        out = np.array([32, 16, 16, 8,4])
                    else:
                        if inp_shape == 131072:
                            out = np.array([16, 16, 16, 8,4])
                        else:
                            if inp_shape == 65536:
                                out = np.array([16, 16, 16, 4,4])
                            else:
                                if inp_shape == 32768:
                                    out = np.array([16, 16, 8, 4,4])
                                else:
                                    if inp_shape == 16384:
                                        out = np.array([16, 8, 8, 4,4])
                                    else:
                                        if inp_shape == 8192:
                                            out = np.array([8, 8, 8, 4,4])
                                        else:
                                            if inp_shape == 4096:
                                                out = np.array([8, 8, 8, 2,4])
                                            else:
                                                if inp_shape == 2048:
                                                    out = np.array([8, 4, 8, 2,4])
                                                else:
                                                    if inp_shape == 1024:
                                                        out = np.array([8, 4, 4, 2,4])
                                                    else:
                                                        if inp_shape == 512:
                                                            out = np.array([8, 4, 4, 2,2])
                                                        else:
                                                            if inp_shape == 256:
                                                                out = np.array([4, 4, 4, 2,2])
                                                            else:
                                                                if inp_shape == 128:
                                                                    out = np.array([4, 2, 4, 2,2])
                                                                else:
                                                                    if inp_shape == 64:
                                                                        out = np.array([4, 2, 2, 2,2])
                                                                    else:
                                                                        if inp_shape == 32:
                                                                            out = np.array([2, 2, 2, 2,2])
                                                                        else:
                                                                            if inp_shape == 16:
                                                                                out = np.array([2, 2, 2, 2,1])
                                                                            else:
                                                                                if inp_shape == 8:
                                                                                    out = np.array([2, 2, 1, 2,1])
                                                                                else:
                                                                                    if inp_shape == 4:
                                                                                        out = np.array([2, 1, 1, 2,1])
        else:
            if split_dimension_core == 6:
                if inp_shape == 1048576:
                    out = np.array([8, 4, 32, 32, 8, 4])
                else:
                    if inp_shape == 524288:
                        out = np.array([8, 4, 32, 16, 8, 4])
                    else:
                        if inp_shape == 262144:
                            out = np.array([8, 4, 16, 16, 8, 4])
                        else:
                            if inp_shape == 131072:
                                out = np.array([4,4, 16, 16, 8, 4])
                            else:
                                if inp_shape == 65536:
                                    out = np.array([4,4, 16, 16, 4, 4])
                                else:
                                    if inp_shape == 32768:
                                        out = np.array([4,4, 16, 8, 4, 4])
                                    else:
                                        if inp_shape == 16384:
                                            out = np.array([4,4, 8, 8, 4, 4])
                                        else:
                                            if inp_shape == 8192:
                                                out = np.array([4,2, 8, 8, 4, 4])
                                            else:
                                                if inp_shape == 4096:
                                                    out = np.array([4,2, 8, 8, 2, 4])
                                                else:
                                                    if inp_shape == 2048:
                                                        out = np.array([4,2, 4, 8, 2, 4])
                                                    else:
                                                        if inp_shape == 1024:
                                                            out = np.array([4,2, 4, 4, 2, 4])
                                                        else:
                                                            if inp_shape == 512:
                                                                out = np.array([4,2, 4, 4, 2, 2])
                                                            else:
                                                                if inp_shape == 256:
                                                                    out = np.array([2, 2, 4, 4, 2, 2])
                                                                else:
                                                                    if inp_shape == 128:
                                                                        out = np.array([2, 2, 2, 4, 2, 2])
                                                                    else:
                                                                        if inp_shape == 64:
                                                                            out = np.array([2, 2, 2, 2, 2, 2])
                                                                        else:
                                                                            if inp_shape == 32:
                                                                                out = np.array([1,2, 2, 2, 2, 2])
                                                                            else:
                                                                                if inp_shape == 16:
                                                                                    out = np.array([1,2, 2, 2, 2, 1])
                                                                                else:
                                                                                    if inp_shape == 8:
                                                                                        out = np.array([1,2, 2, 1, 2, 1])
                                                                                    else:
                                                                                        if inp_shape == 4:
                                                                                            out = np.array(
                                                                                                [1,2, 1, 1, 2, 1])
            else:
                if split_dimension_core == 7:
                    if inp_shape == 1048576:
                        out = np.array([8, 4, 8, 4, 32, 8, 4])
                    else:
                        if inp_shape == 524288:
                            out = np.array([8, 4, 8, 4, 16, 8, 4])
                        else:
                            if inp_shape == 262144:
                                out = np.array([8, 4, 4, 4, 16, 8, 4])
                            else:
                                if inp_shape == 131072:
                                    out = np.array([4, 4, 4, 4, 16, 8, 4])
                                else:
                                    if inp_shape == 65536:
                                        out = np.array([4, 4, 4, 4, 16, 4, 4])
                                    else:
                                        if inp_shape == 32768:
                                            out = np.array([4, 4, 4, 4, 8, 4, 4])
                                        else:
                                            if inp_shape == 16384:
                                                out = np.array([4, 4, 4, 2, 8, 4, 4])
                                            else:
                                                if inp_shape == 8192:
                                                    out = np.array([4, 2, 4, 2, 8, 4, 4])
                                                else:
                                                    if inp_shape == 4096:
                                                        out = np.array([4, 2, 4, 2, 8, 2, 4])
                                                    else:
                                                        if inp_shape == 2048:
                                                            out = np.array([4, 2, 4, 4, 2, 2, 4])
                                                        else:
                                                            if inp_shape == 1024:
                                                                out = np.array([4, 2, 4, 2, 2, 2, 4])
                                                            else:
                                                                if inp_shape == 512:
                                                                    out = np.array([4, 2, 4, 2, 2, 2, 2])
                                                                else:
                                                                    if inp_shape == 256:
                                                                        out = np.array([2, 2, 2, 2, 4, 2, 2])
                                                                    else:
                                                                        if inp_shape == 128:
                                                                            out = np.array([2, 2, 2, 2, 2, 2, 2])
                                                                        else:
                                                                            if inp_shape == 64:
                                                                                out = np.array([1,2, 2, 2, 2, 2, 2])
                                                                            else:
                                                                                if inp_shape == 32:
                                                                                    out = np.array([1, 2, 2, 2, 2, 2,1])
                                                                                else:
                                                                                    if inp_shape == 16:
                                                                                        out = np.array(
                                                                                            [1, 2, 2, 2, 2, 1,1])
                                                                                    else:
                                                                                        if inp_shape == 8:
                                                                                            out = np.array(
                                                                                                [1, 2, 2, 1, 2, 1,1])
                                                                                        else:
                                                                                            if inp_shape == 4:
                                                                                                out = np.array(
                                                                                                    [1, 2, 1, 1, 2, 1,1])
                else:
                    if split_dimension_core == 8:
                        if inp_shape == 1048576:
                            out = np.array([8, 4, 8, 4, 8, 4, 8, 4])
                        else:
                            if inp_shape == 524288:
                                out = np.array([8, 4, 8, 4, 2,8, 8, 4])
                            else:
                                if inp_shape == 262144:
                                    out = np.array([8, 4, 4, 4, 2,8, 8, 4])
                                else:
                                    if inp_shape == 131072:
                                        out = np.array([4, 4, 4, 4, 2,8, 8, 4])
                                    else:
                                        if inp_shape == 65536:
                                            out = np.array([4, 4, 4, 4, 2,8, 4, 4])
                                        else:
                                            if inp_shape == 32768:
                                                out = np.array([4, 4, 4, 4, 2,4, 4, 4])
                                            else:
                                                if inp_shape == 16384:
                                                    out = np.array([4, 4, 4, 2, 2,4, 4, 4])
                                                else:
                                                    if inp_shape == 8192:
                                                        out = np.array([4, 2, 4, 2, 2,4, 4, 4])
                                                    else:
                                                        if inp_shape == 4096:
                                                            out = np.array([4, 2, 4, 2, 2,4, 2, 4])
                                                        else:
                                                            if inp_shape == 2048:
                                                                out = np.array([4, 2, 4, 2, 2, 2, 2, 4])
                                                            else:
                                                                if inp_shape == 1024:
                                                                    out = np.array([4, 2, 2, 2, 2, 2, 2, 4])
                                                                else:
                                                                    if inp_shape == 512:
                                                                        out = np.array([4, 2, 2, 2, 2, 2, 2, 2])
                                                                    else:
                                                                        if inp_shape == 256:
                                                                            out = np.array([2, 2, 2, 2, 2, 2, 2, 2])
                                                                        else:
                                                                            if inp_shape == 128:
                                                                                out = np.array([1,2, 2, 2, 2, 2, 2, 2])
                                                                            else:
                                                                                if inp_shape == 64:
                                                                                    out = np.array(
                                                                                        [1,1, 2, 2, 2, 2, 2, 2])
                                                                                else:
                                                                                    if inp_shape == 32:
                                                                                        out = np.array(
                                                                                            [1,1, 2, 2, 2, 2, 2, 1])
                                                                                    else:
                                                                                        if inp_shape == 16:
                                                                                            out = np.array(
                                                                                                [1,1, 2, 2, 2, 2, 1, 1])
                                                                                        else:
                                                                                            if inp_shape == 8:
                                                                                                out = np.array(
                                                                                                    [1,1, 2, 2, 1, 2, 1,
                                                                                                     1])
                                                                                            else:
                                                                                                if inp_shape == 4:
                                                                                                    out = np.array(
                                                                                                        [1,1, 2, 1, 1, 2,
                                                                                                         1, 1])
                    else:
                        if split_dimension_core == 9:
                            if inp_shape == 1048576:
                                out = np.array([8, 4, 4, 2, 4, 8, 4, 8, 4])
                            else:
                                if inp_shape == 524288:
                                    out = np.array([8, 4, 4, 2, 4, 2, 8, 8, 4])
                                else:
                                    if inp_shape == 262144:
                                        out = np.array([8, 4, 4, 4, 2, 4, 2, 8, 4])
                                    else:
                                        if inp_shape == 131072:
                                            out = np.array([4, 4, 4, 4, 2, 4, 2, 8, 4])
                                        else:
                                            if inp_shape == 65536:
                                                out = np.array([4, 4, 4, 4, 2, 4, 2, 4, 4])
                                            else:
                                                if inp_shape == 32768:
                                                    out = np.array([4, 4, 4, 2, 2, 2, 4, 4, 4])
                                                else:
                                                    if inp_shape == 16384:
                                                        out = np.array([4, 4, 2, 2, 2, 2, 4, 4, 4])
                                                    else:
                                                        if inp_shape == 8192:
                                                            out = np.array([4, 2, 2, 2, 2, 2, 4, 4, 4])
                                                        else:
                                                            if inp_shape == 4096:
                                                                out = np.array([4, 2, 2, 2, 2, 2, 4, 2, 4])
                                                            else:
                                                                if inp_shape == 2048:
                                                                    out = np.array([4, 2, 2, 2, 2, 2, 2, 2, 4])
                                                                else:
                                                                    if inp_shape == 1024:
                                                                        out = np.array([2, 2, 2, 2, 2, 2, 2, 2, 4])
                                                                    else:
                                                                        if inp_shape == 512:
                                                                            out = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
                                                                        else:
                                                                            if inp_shape == 256:
                                                                                out = np.array([1,2, 2, 2, 2, 2, 2, 2, 2])
                                                                            else:
                                                                                if inp_shape == 128:
                                                                                    out = np.array(
                                                                                        [1,1, 2, 2, 2, 2, 2, 2, 2])
                                                                                else:
                                                                                    if inp_shape == 64:
                                                                                        out = np.array(
                                                                                            [1,1, 1, 2, 2, 2, 2, 2, 2])
                                                                                    else:
                                                                                        if inp_shape == 32:
                                                                                            out = np.array(
                                                                                                [1,1, 1, 2, 2, 2, 2, 2,
                                                                                                 1])
                                                                                        else:
                                                                                            if inp_shape == 16:
                                                                                                out = np.array(
                                                                                                    [1,1, 1, 2, 2, 2, 2,
                                                                                                     1, 1])
                                                                                            else:
                                                                                                if inp_shape == 8:
                                                                                                    out = np.array(
                                                                                                        [1,1, 1, 2, 2, 1,
                                                                                                         2, 1,
                                                                                                         1])
                                                                                                else:
                                                                                                    if inp_shape == 4:
                                                                                                        out = np.array(
                                                                                                            [1,1, 1, 2, 1,
                                                                                                             1, 2,
                                                                                                             1, 1])
                        else:
                            if split_dimension_core == 10:
                                if inp_shape == 1048576:
                                    out = np.array([8, 4, 4, 2, 4, 4, 2, 4, 8, 4])
                                else:
                                    if inp_shape == 524288:
                                        out = np.array([8, 4, 4, 2, 4, 2, 4, 2, 8, 4])
                                    else:
                                        if inp_shape == 262144:
                                            out = np.array([8, 4, 4, 4, 2, 4, 2, 4, 2, 4])
                                        else:
                                            if inp_shape == 131072:
                                                out = np.array([4, 4, 4, 4, 2, 4, 2, 4, 2, 4])
                                            else:
                                                if inp_shape == 65536:
                                                    out = np.array([4, 4, 4, 4, 2, 2, 2, 2, 4, 4])
                                                else:
                                                    if inp_shape == 32768:
                                                        out = np.array([4, 4, 2, 2, 2, 2, 2, 4, 4, 4])
                                                    else:
                                                        if inp_shape == 16384:
                                                            out = np.array([4, 4, 2, 2, 2, 2, 2, 2, 4, 4])
                                                        else:
                                                            if inp_shape == 8192:
                                                                out = np.array([4, 2, 2, 2, 2, 2, 2, 2, 4, 4])
                                                            else:
                                                                if inp_shape == 4096:
                                                                    out = np.array([4, 2, 2, 2, 2, 2, 2, 2, 2, 4])
                                                                else:
                                                                    if inp_shape == 2048:
                                                                        out = np.array([4, 2, 2, 2, 2, 2, 2, 2, 2, 2])
                                                                    else:
                                                                        if inp_shape == 1024:
                                                                            out = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
                                                                        else:
                                                                            if inp_shape == 512:
                                                                                out = np.array(
                                                                                    [2, 2, 2, 2, 2, 2, 2, 1,2, 2])
                                                                            else:
                                                                                if inp_shape == 256:
                                                                                    out = np.array(
                                                                                        [1, 2, 2, 2, 2, 1,2, 2, 2, 2])
                                                                                else:
                                                                                    if inp_shape == 128:
                                                                                        out = np.array(
                                                                                            [1, 1, 2, 2, 2, 1,2, 2, 2, 2])
                                                                                    else:
                                                                                        if inp_shape == 64:
                                                                                            out = np.array(
                                                                                                [1, 1, 1, 2,1, 2, 2, 2, 2,
                                                                                                 2])
                                                                                        else:
                                                                                            if inp_shape == 32:
                                                                                                out = np.array(
                                                                                                    [1, 1, 1, 2, 2,1, 2,
                                                                                                     2, 2,
                                                                                                     1])
                                                                                            else:
                                                                                                if inp_shape == 16:
                                                                                                    out = np.array(
                                                                                                        [1, 1, 1, 2, 1,2,
                                                                                                         2, 2,
                                                                                                         1, 1])
                                                                                                else:
                                                                                                    if inp_shape == 8:
                                                                                                        out = np.array(
                                                                                                            [1, 1, 1, 2,
                                                                                                             1,2, 1,
                                                                                                             2, 1,
                                                                                                             1])
                                                                                                    else:
                                                                                                        if inp_shape == 4:
                                                                                                            out = np.array(
                                                                                                                [1, 1,
                                                                                                                 1, 2,1,
                                                                                                                 1,
                                                                                                                 1, 2,
                                                                                                                 1, 1])











    out.astype(np.int32)
    if len(out)==split_dimension_core:
        return out





def lrelu(x, leak=0.2, name="lrelu"):
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
       activation_fn=lrelu, name="fcTensorNet", split_dimension_core=10,tt_rank=50):
    with tf.variable_scope(name):
        inp_shape = input.get_shape().as_list()[0]
        inp = tf.reshape(input, [inp_shape, -1])
        inp_shape_split = inp.get_shape().as_list()[1]
        input_mode=split_dimension_to_rank_multiple(inp_shape_split,split_dimension_core)
        out_mode = split_dimension_to_rank_multiple(output_shape, split_dimension_core)
        tt_rank_array = np.array([1], dtype=np.int32)
        for ix in range(split_dimension_core):
            if ix==split_dimension_core-1:
                tt_rank_array =np.append(tt_rank_array, [1])
            else:
                tt_rank_array = np.append(tt_rank_array, [tt_rank])


        _ = tensornet.layers.tt(inp, input_mode, out_mode,  tt_rank_array ,
                                biases_initializer=None, scope=name)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
        result=tf.reshape(_, [inp_shape, 1, 1, -1])
    return result
