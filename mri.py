from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from datasets.hdf5_loader import create_default_splits as create_default_splits
import tensorflow as tf
import mri_input
import numpy as np
import tensorflow as tf
from ops import conv3d_denseNet
from ops import add_block
from util import log
from ops import depthwise_conv2d
from ops import fc
from ops import transition_layer
from ops import transition_layer_to_classes
from ops import grouped_conv2d_Discriminator_one
from ops import conv3d_denseNet_first_layer
from datasets.hdf5_loader import get_data




NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mri_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = mri_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'



def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(config,whichFoldData):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  dataset_path = os.path.join(r"/data1/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/")

  dataset_train, dataset_test, all_hdf5_data = create_default_splits(dataset_path, hdf5FileName=config.hdf5FileName,
                                                              idFileName=config.idFileName,
                                                              cross_validation_number=config.cross_validation_number)
  # dataset_train, dataset_test are 10 cross validation data.
  # dataset_train[i] is the i-th fold data

  img, label = get_data(dataset_train[0][0],all_hdf5_data)

  print("step3")
  config.h = img.shape[0]
  config.w = img.shape[1]
  config.c = img.shape[2]

  # if len(img.shape) == 3:
  #     config.c = img.shape[2]
  # else:
  #     config.c = 1

  config.num_class = label.shape[0]
  nput_ops, batch_train = mri_input.distorted_inputs(all_hdf5_data, whichFoldData,dataset_train,
                                                  config.batch_size)

    # images = tf.cast(images, tf.float16)
    # labels = tf.cast(labels, tf.float16)
  images, labels = batch_train['image'], batch_train['label']
  return images, labels, len(dataset_train[0]), dataset_test, all_hdf5_data


def distorted_inputs_test(all_hdf5_data,dataset_test,batch_size, which_fold_data):

  input_ops, batch_train = mri_input.distorted_inputs_test(all_hdf5_data,dataset_test,batch_size,which_fold_data)

  ids, images, labels = batch_train['id'], batch_train['image'], batch_train['label']
  return ids,images, labels




def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not config.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(config.data_dir, 'cifar-10-batches-bin')
  images, labels = mri_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=config.batch_size)
  if config.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images,config):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  first_output_features = config.growth_rate * 2
  total_blocks = config.total_blocks
  layers_per_block = (config.depth - (total_blocks + 1)) // total_blocks
  bc_mode = config.bc_mode
  # compression rate at the transition layers
  reduction = config.reduction
  keep_prob = config.keep_prob
  model_type = config.model_type
  growth_rate = config.growth_rate
  _is_train = True
  _num_class = config.num_class
  # first - initial 3 x 3 conv to first_output_features
  if not bc_mode:
      print("Build %s model with %d blocks, "
            "%d composite layers each." % (
                model_type, total_blocks, layers_per_block))
  if bc_mode:
      layers_per_block = layers_per_block // 2
      print("Build %s model with %d blocks, "
            "%d bottleneck layers and %d composite layers each." % (
                model_type, total_blocks, layers_per_block,
                layers_per_block))
  print("Reduction at transition layers: %.1f" % reduction)






  with tf.variable_scope("Initial_convolution"):
      output = conv3d_denseNet_first_layer(
          images,
          out_features=first_output_features,
          kernel_size=3)

  # add N required blocks
  for block in range(total_blocks):
      with tf.variable_scope("Block_%d" % block):
          output = add_block(keep_prob, _is_train, output, growth_rate, layers_per_block, bc_mode)
      # last block exist without transition layer
      if block != total_blocks - 1:
          with tf.variable_scope("Transition_after_block_%d" % block):
              output = transition_layer(output, _is_train, keep_prob, reduction)

  with tf.variable_scope("Transition_to_classes"):
      softmax_linear = transition_layer_to_classes(output, _num_class, _is_train)

  # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
  return softmax_linear


def loss(logits, labels, config):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_per_batch')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).

  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  l2_loss = tf.add_n(
      [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

  loss_l2_regular = total_loss + l2_loss * config.weight_decay

  prediction = tf.nn.softmax(logits)

  tf.add_to_collection('prediction_label', prediction)
  tf.add_to_collection('target_label', labels)

  correct_prediction_each_batch = tf.equal(
      tf.argmax(prediction, 1),
      tf.argmax(labels, 1))
  accuracy_each_batch = tf.reduce_mean(tf.cast(correct_prediction_each_batch, tf.float32))






  return loss_l2_regular, accuracy_each_batch,prediction,labels


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op



