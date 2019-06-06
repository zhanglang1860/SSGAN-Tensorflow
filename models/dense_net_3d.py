import os
import time
import shutil
import platform
from datetime import timedelta

import numpy as np
import tensorflow as tf
from util import log
import tensorflow.contrib.slim as slim

import csv
TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))



class EvalManager(object):

  def __init__(self,train_dir):
    # collection of batches (not flattened)
    self._ids = []
    self._predictions = []
    self._groundtruths = []
    self.train_dir=train_dir



  # def add_batch(self, id, prediction, groundtruth):
  #
  #     # for now, store them all (as a list of minibatch chunks)
  #     self._ids.append(id)
  #     self._predictions.append(prediction)
  #     self._groundtruths.append(groundtruth)

  def compute_accuracy(self, pred, gt):
    correct_prediction = np.sum(np.argmax(pred[:, :-1], axis=1) == np.argmax(gt, axis=1))
    return float(correct_prediction) / pred.shape[0]

  def add_batch_new(self, prediction, groundtruth):

    # for now, store them all (as a list of minibatch chunks)
    shold_be_batch_size = len(prediction)
    for index in range(shold_be_batch_size):
      self._predictions.append(prediction[index])
      self._groundtruths.append(groundtruth[index])

  def add_batch(self, id, prediction, groundtruth):

    # for now, store them all (as a list of minibatch chunks)
    shold_be_batch_size = len(id)
    for index in range(shold_be_batch_size):
      self._ids.append(id[index])
      self._predictions.append(prediction[index])
      self._groundtruths.append(groundtruth[index])

  def report(self, result_file_name):
    # report L2 loss
    # log.info("Computing scores...")

    z = zip(self._predictions, self._groundtruths)
    u = list(z)

    if tf.gfile.Exists(self.train_dir + '/GANresults/'):
      log.infov("self.train_dir + '/GANresults/' exists")
    else:
      os.makedirs(self.train_dir + '/GANresults/')

    with open(self.train_dir + '/GANresults/' + result_file_name + '.csv', mode='w') as file:
      writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['actual_label', 'model_GAN'])
      for pred, gt in u:
        gt_csv = np.argmax(gt)
        pred_csv = np.argmax(pred)

        one_row = []

        one_row.append(str(gt_csv))
        one_row.append(str(pred_csv))

        writer.writerow(one_row)




class GAN3D(object):
  def __init__(self, config,data_provider,all_train_dir,whichFoldData,is_train=True):
    """
    Class to implement networks base on this paper
    https://arxiv.org/pdf/1611.05552.pdf

    Args:
      data_provider: Class, that have all required data sets
      growth_rate: `int`, variable from paper
      depth: `int`, variable from paper
      total_blocks: `int`, paper value == 3
      keep_prob: `float`, keep probability for dropout. If keep_prob = 1
        dropout will be disables
      weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
      nesterov_momentum: `float`, momentum for Nesterov optimizer
      model_type: `str`, 'DenseNet3D' or 'DenseNet3D-BC'. Should model use
        bottle neck connections or not.
      dataset: `str`, dataset name
      should_save_logs: `bool`, should logs be saved or not
      should_save_model: `bool`, should model be saved or not
      renew_logs: `bool`, remove previous logs for current model
      reduction: `float`, reduction Theta at transition layer for
        DenseNets with bottleneck layers. See paragraph 'Compression'
        https://arxiv.org/pdf/1608.06993v3.pdf#4
      bc_mode: `bool`, should we use bottleneck layers and features
        reduction or not.
    """
    tf.reset_default_graph()
    self.data_provider          = data_provider
    self.data_shape             = data_provider.data_shape
    self.n_classes              = data_provider.n_classes
    self.depth                  = config.depth
    self.growth_rate            = config.growth_rate
    # how many features will be received after first convolution
    # value the same as in the original Torch code
    self.first_output_features  = config.growth_rate * 2
    self.total_blocks           = config.total_blocks
    self.layers_per_block       = (config.depth - (config.total_blocks + 1)) // config.total_blocks
    self.bc_mode = config.bc_mode
    # compression rate at the transition layers
    self.reduction              = config.reduction
    if not config.bc_mode:
      print("Build %s model with %d blocks, "
          "%d composite layers each." % (
            config.model_type, self.total_blocks, self.layers_per_block))
    if config.bc_mode:
      self.layers_per_block     = self.layers_per_block // 2
      print("Build %s model with %d blocks, "
          "%d bottleneck layers and %d composite layers each." % (
            config.model_type, self.total_blocks, self.layers_per_block,
            self.layers_per_block))
    print("Reduction at transition layers: %.1f" % self.reduction)

    self.keep_prob          = config.keep_prob
    self.weight_decay       = config.weight_decay
    self.nesterov_momentum  = config.nesterov_momentum
    self.model_type         = config.model_type
    self.dataset_name       = config.hdf5FileNametrain

    self.batches_step       = 0
    self.sequence_length    = 109
    self.crop_size          = (91,91)
    self.train_dir= all_train_dir[whichFoldData]
    self.whichFoldData=whichFoldData
    self.renew_logs=config.renew_logs
    self.n_z = config.n_z
    self.batch_size = config.batch_size
    self.d_thresh = config.d_thresh
    self.weights = self.initialiseWeights()
    self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

    self.build_GAN(is_train=is_train)



    self._initialize_session()



    # self._count_trainable_params()

  def build_GAN(self, is_train=True):

    self._define_inputs_D_G()

    fake_image = self._build_graph_G(self.z_vector,phase_train=is_train)
    d_real, d_no_sigmoid_real=self._build_graph_D(self.x_vector, reuse=False)

    d_real = tf.maximum(tf.minimum(d_real, 0.99), 0.01)


    self.summary_d_x_hist = tf.summary.histogram("d_prob_x", d_real)

    d_fake, d_no_sigmoid_fake = self._build_graph_D(fake_image, reuse=True)

    d_fake = tf.maximum(tf.minimum(d_fake, 0.99), 0.01)
    self.summary_d_z_hist = tf.summary.histogram("d_prob_z", d_fake)

    # Compute the discriminator accuracy
    n_p_x = tf.reduce_sum(tf.cast(d_real > 0.5, tf.int32))
    n_p_z = tf.reduce_sum(tf.cast(d_fake < 0.5, tf.int32))
    d_acc = tf.divide(n_p_x + n_p_z, 2 * self.batch_size)

    # Compute the discriminator and generator loss
    # d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1-d_fake))
    # g_loss = -tf.reduce_mean(tf.log(d_fake))
    n=self.n_classes

    def build_loss(d_real, d_real_logits, d_fake, d_fake_logits, label, real_image, fake_image):
      alpha = 0.9
      real_label = tf.concat([label, tf.zeros([self.batch_size, 1])], axis=1)
      fake_label = tf.concat([(1 - alpha) * tf.ones([self.batch_size, n]) / n,
                              alpha * tf.ones([self.batch_size, 1])], axis=1)

      # Discriminator/classifier loss
      s_loss = tf.reduce_mean(self.huber_loss(label, d_real[:, :-1]))
      d_loss_real = tf.nn.softmax_cross_entropy_with_logits(
        logits=d_real_logits, labels=real_label)
      d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(
        logits=d_fake_logits, labels=fake_label)
      d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

      # Generator loss
      # g_loss = tf.reduce_mean(tf.log(d_fake[:, -1]))

      g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_fake[:, -1], labels=tf.ones_like(d_fake[:, -1]))

      d_loss = tf.reduce_mean(self.d_loss)
      g_loss = tf.reduce_mean(self.g_loss)



      # # Weight annealing
      # g_loss += tf.reduce_mean(
      #   self.huber_loss(real_image, fake_image)) * self.recon_weight

      GAN_loss = tf.reduce_mean(d_loss + g_loss)

      # Classification accuracy
      correct_prediction = tf.equal(tf.argmax(d_real[:, :-1], 1),
                                    tf.argmax(self.label, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      return s_loss, d_loss_real, d_loss_fake, d_loss, g_loss, GAN_loss, accuracy




    self.all_preds = d_real
    self.all_targets = self.labels

    self.S_loss, d_loss_real, d_loss_fake, self.d_loss, self.g_loss, GAN_loss, self.accuracy = \
      build_loss(d_real, d_no_sigmoid_real, d_fake, d_no_sigmoid_fake, self.labels, self.x_vector, fake_image)





    self.summary_d_loss = tf.summary.scalar("d_loss", self.d_loss)
    self.summary_g_loss = tf.summary.scalar("g_loss", self.g_loss)
    self.summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    self.summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    self.summary_d_acc = tf.summary.scalar("d_acc", self.accuracy)

    # net_g_test = self._build_graph_G(self.z_vector, phase_train=False, reuse=True)

    para_g = [v for v in tf.trainable_variables() if v.name.startswith('Generator')]
    para_d = [v for v in tf.trainable_variables() if v.name.startswith('Discriminator')]

    # only update the weights for the discriminator network
    self.optimizer_op_d = tf.train.MomentumOptimizer(
      self.learning_rate, self.nesterov_momentum, use_nesterov=True).minimize(self.d_loss, var_list=para_d)
    # only update the weights for the generator network
    self.optimizer_op_g = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5).minimize(self.g_loss, var_list=para_g)




  def initialiseWeights(self):

    weights = {}
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, 128], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 32, 64], initializer=xavier_init)
    weights['wg6'] = tf.get_variable("wg6", shape=[4, 4, 4, 1, 32], initializer=xavier_init)

    return weights

  def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)



  def _initialize_session(self):
    """Initialize session, variables, saver"""

    config = tf.ConfigProto()
    # restrict model GPU memory utilization to min required
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    tf_ver = int(tf.__version__.split('.')[1])
    if TF_VERSION <= 0.10:
      self.sess.run(tf.initialize_all_variables())
      logswriter = tf.summary.SummaryWriter(self.train_dir)
    else:
      self.sess.run(tf.global_variables_initializer())
      logswriter = tf.summary.FileWriter
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
    self.summary_writer = logswriter(self.train_dir, self.sess.graph)
    self.summary_op = tf.summary.merge_all()


  # (Updated)
  def _count_trainable_params(self):
    total_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parametes = 1
      for dim in shape:
        variable_parametes *= dim.value
      total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

  @property
  def save_path(self):
    try:
      save_path = self._save_path
      model_path = self._model_path
    except AttributeError:
      save_path = self.train_dir
      if platform.python_version_tuple()[0] is '2':
        if not os.path.exists(save_path):
          os.makedirs(save_path)
      else:
        os.makedirs(save_path, exist_ok=True)
      model_path = os.path.join(save_path, 'model.chkpt')
      self._save_path = save_path
      self._model_path = model_path
    return save_path, model_path



  @property
  def model_identifier(self):
    return "{}_growth_rate={}_depth={}_seq_length={}_crop_size={}".format(
      self.model_type, self.growth_rate, self.depth, self.sequence_length,
      self.crop_size)

  # (Updated)
  def save_model(self, global_step=None):
    self.saver.save(self.sess, self.train_dir+'/', global_step=global_step)

  def load_model(self):
    """load the sess from the pretrain model

      Returns:
        start_epoch: the start step to train the model
    """
    # Restore the trianing model from the folder
    ckpt =  tf.train.get_checkpoint_state(self.save_path[0])
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      start_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      start_epoch = int(start_epoch) + 1
      print("Successfully load model from save path: %s and epoch: %s" 
          % (self.save_path[0], start_epoch))
      return start_epoch
    else:
      print("Training from scratch")
      return 1

  # (Updated)
  def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                        should_print=True):
    print("mean cross_entropy: %f, mean accuracy: %f" % (
        loss, accuracy))
    summary = tf.Summary(value=[
      tf.Summary.Value(
        tag='loss_%s' % prefix, simple_value=float(loss)),
      tf.Summary.Value(
        tag='accuracy_%s' % prefix, simple_value=float(accuracy))
    ])
    self.summary_writer.add_summary(summary, epoch)

  # (Updated)
  def _define_inputs_D_G(self):
    shape = [None]
    shape.extend(self.data_shape)
    # self.videos = tf.placeholder(
    #   tf.float32,
    #   shape=shape,
    #   name='input_videos')
    self.labels = tf.placeholder(
      tf.float32,
      shape=[None, self.n_classes],
      name='labels')
    self.learning_rate = tf.placeholder(
      tf.float32,
      shape=[],
      name='learning_rate')
    self.x_vector = tf.placeholder(shape=shape, dtype=tf.float32)
    self.z_vector = tf.placeholder(shape=[None, self.n_z], dtype=tf.float32)



    # (Updated)
  def composite_function(self, _input, out_features, kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
      # BN
      output = self.batch_norm(_input)
      # ReLU
      with tf.name_scope("ReLU"):
        output = tf.nn.relu(output)
      # convolution
      output = self.conv3d(
        output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
      output = self.dropout(output)
    return output

  # (Updated)
  def bottleneck(self, _input, out_features):
    with tf.variable_scope("bottleneck"):
      output = self.batch_norm(_input)
      with tf.name_scope("ReLU"):
        output = tf.nn.relu(output)
      inter_features = out_features * 4
      output = self.conv3d(
        output, out_features=inter_features, kernel_size=1,
        padding='VALID')
      output = self.dropout(output)
    return output

  # (Updated)
  def add_internal_layer(self, _input, growth_rate):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not self.bc_mode:
      comp_out = self.composite_function(
        _input, out_features=growth_rate, kernel_size=3)
    elif self.bc_mode:
      bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
      comp_out = self.composite_function(
        bottleneck_out, out_features=growth_rate, kernel_size=3)
    # concatenate _input with out from composite function
    with tf.name_scope("concat"):
      if TF_VERSION >= 1.0:
          output = tf.concat(axis=4, values=(_input, comp_out))
      else:
        output = tf.concat(4, (_input, comp_out))
    return output

  # (Updated)
  def add_block(self, _input, growth_rate, layers_per_block):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
      with tf.variable_scope("layer_%d" % layer):
        output = self.add_internal_layer(output, growth_rate)
    return output

  # (Updated)
  def transition_layer(self, _input, pool_depth=2):
    """Call H_l composite function with 1x1 kernel and pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * self.reduction)
    output = self.composite_function(
      _input, out_features=out_features, kernel_size=1)
    # run pooling
    with tf.name_scope("pooling"):
      output = self.pool(output, k=2, d=pool_depth)
    return output

  # (Updated)
  def trainsition_layer_to_classes(self, _input):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide pooling
    - FC layer multiplication
    """
    # BN
    output = self.batch_norm(_input)
    # ReLU
    with tf.name_scope("ReLU"):
      output = tf.nn.relu(output)
    # pooling
    last_pool_kernel_width = int(output.get_shape()[-2])
    last_pool_kernel_height = int(output.get_shape()[-3])
    last_sequence_length = int(output.get_shape()[1])
    with tf.name_scope("pooling"):
      output = self.pool(output, k = last_pool_kernel_height,
                         d = last_sequence_length,
                         width_k = last_pool_kernel_width,
                         k_stride_width = last_pool_kernel_width)
    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    W = self.weight_variable_xavier(
      [features_total, self.n_classes], name='W')
    bias = self.bias_variable([self.n_classes])
    logits = tf.matmul(output, W) + bias
    # Local 
    # features_total = int(output.get_shape()[-1])
    # output = tf.reshape(output, [-1, features_total])
    # lc_weight = self.weight_variable_xavier(
    #   [features_total, 100], name='lc_weight')
    # lc_bias = self.bias_variable([100], 'lc_bias')
    # local = tf.nn.relu(tf.matmul(output, lc_weight) + lc_bias)
    # local = self.dropout(local)
    # # Second Local
    # lc2_weight = self.weight_variable_xavier(
    #   [100, 100], name='lc2_weight')
    # lc2_bias = self.bias_variable([100], 'lc2_bias')
    # local2 = tf.nn.relu(tf.matmul(local, lc2_weight) + lc2_bias)
    # local2 = self.dropout(local2)
    # # Classification
    # weight = self.weight_variable_xavier(
    #   [100, self.n_classes], name='weight')
    # bias = self.bias_variable([self.n_classes], 'bias')
    # logits = tf.matmul(local2, weight) + bias
    return logits
  
  # (Updated)
  def conv3d(self, _input, out_features, kernel_size,
         strides=[1, 1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = self.weight_variable_msra(
      [kernel_size, kernel_size, kernel_size, in_features, out_features],
      name='kernel')
    with tf.name_scope("3DConv"):
      output = tf.nn.conv3d(_input, kernel, strides, padding)
    return output

  # (Updated)
  def pool(self, _input, k, d=2, width_k=None, type='avg', k_stride=None, d_stride=None, k_stride_width=None):
    if not width_k: width_k = k
    ksize = [1, d, k, width_k, 1]
    if not k_stride: k_stride = k
    if not k_stride_width: k_stride_width = k_stride
    if not d_stride: d_stride = d
    strides = [1, d_stride, k_stride, k_stride_width, 1]
    padding = 'SAME'
    if type is 'max':
      output = tf.nn.max_pool3d(_input, ksize, strides, padding)
    elif type is 'avg':
      output = tf.nn.avg_pool3d(_input, ksize, strides, padding)
    else:
      output = None
    return output

  # (Updated)
  def batch_norm(self, _input):
    with tf.name_scope("batch_normalization"):
      output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=self.is_training,
        updates_collections=None)
    return output

  # (Updated)
  def dropout(self, _input):
    if self.keep_prob < 1:
      with tf.name_scope('dropout'):
        output = tf.cond(
          self.is_training,
          lambda: tf.nn.dropout(_input, self.keep_prob),
          lambda: _input
        )
    else:
      output = _input
    return output

  # (Updated)
  def weight_variable_msra(self, shape, name):
    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=tf.contrib.layers.variance_scaling_initializer())

  # (Updated)
  def weight_variable_xavier(self, shape, name):
    return tf.get_variable(
      name,
      shape=shape,
      initializer=tf.contrib.layers.xavier_initializer())


  def bias_variable(self, shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


  def _build_graph_G(self, z, phase_train=True):
    strides = [1, 2, 2, 2, 1]
    weights=self.weights
    batch_size=self.batch_size


    with tf.variable_scope("Generator"):
      z = tf.reshape(z, (self.batch_size, 1, 1, 1, self.n_z))
      g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size, 4, 3, 3, 512), strides=[1, 1, 1, 1, 1],
                                   padding="VALID")
      g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
      g_1 = tf.nn.relu(g_1)

      g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size, 7, 6, 6, 256), strides=strides, padding="SAME")
      g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
      g_2 = tf.nn.relu(g_2)

      g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size, 14, 12, 12, 128), strides=strides, padding="SAME")
      g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
      g_3 = tf.nn.relu(g_3)

      g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size, 28, 23, 23, 64), strides=strides, padding="SAME")
      g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
      g_4 = tf.nn.relu(g_4)

      g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size, 55, 46, 46, 32), strides=strides, padding="SAME")
      g_5 = tf.contrib.layers.batch_norm(g_5, is_training=phase_train)
      g_5 = tf.nn.relu(g_5)

      g_6 = tf.nn.conv3d_transpose(g_5, weights['wg6'], (batch_size, 109, 91, 91, 1), strides=strides, padding="SAME")

      g_6 = tf.nn.tanh(g_6)

    print g_1, 'g1'
    print g_2, 'g2'
    print g_3, 'g3'
    print g_4, 'g4'
    print g_5, 'g5'
    print g_6, 'g6'

    return g_6





  def _build_graph_D(self,mri,reuse=False):
    growth_rate = self.growth_rate
    layers_per_block = self.layers_per_block
    # first - initial 3 x 3 x 3 conv to first_output_features


    with tf.variable_scope("Discriminator", reuse=reuse):
      with tf.variable_scope("Initial_convolution"):
        output = self.conv3d(
          mri,
          out_features=self.first_output_features,
          kernel_size=7,
          strides=[1, 1, 2, 2, 1])
        # first pooling
        output = self.pool(output, k=3, d=3, k_stride=2, d_stride=1)

      # add N required blocks
      for block in range(self.total_blocks):
        with tf.variable_scope("Block_%d" % block):
          output = self.add_block(output, growth_rate, layers_per_block)
        # last block exist without transition layer
        if block != self.total_blocks - 1:
          with tf.variable_scope("Transition_after_block_%d" % block):
            # pool_depth = 1 if block == 0 else 2
            pool_depth = 2
            output = self.transition_layer(output, pool_depth)

      with tf.variable_scope("Transition_to_classes"):
        logits = self.trainsition_layer_to_classes(output)

    logits_sigmoid = tf.nn.sigmoid(logits)
    return logits_sigmoid, logits


  # (Updated)
  def train_all_epochs(self, config):
    n_epochs           = config.max_training_steps
    init_learning_rate = config.learning_rate_d
    batch_size         = config.batch_size
    reduce_lr_epoch_1  = config.reduce_lr_epoch_1
    reduce_lr_epoch_2  = config.reduce_lr_epoch_2
    total_start_time   = time.time()

    # Restore the model if we have
    start_epoch = self.load_model()
    # to illustrate overfitting with accuracy and loss later
    # f = open(self.train_dir +'/accuracy.txt', 'w')
    # f.write('epoch, train_acc, test_acc\n')
    #
    # fx = open(self.train_dir +'/loss.txt', 'w')
    # fx.write('epoch, train_loss, test_loss\n')

    
    # Start training 
    for epoch in range(start_epoch, n_epochs + 1):
      print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
      start_time = time.time()
      learning_rate = init_learning_rate
      # Update the learning rate according to the decay parameter
      if epoch >= reduce_lr_epoch_1 and epoch < reduce_lr_epoch_2:
        learning_rate = learning_rate / 10
        print("Decrease learning rate, new lr = %f" % learning_rate)
      elif epoch >= reduce_lr_epoch_2:
        learning_rate = learning_rate / 100
        print("Decrease learning rate, new lr = %f" % learning_rate)

      print("Training...")



      mean_D_loss, mean_G_loss, acc, lr_value, summary_d, summary_g = self.train_one_epoch(
        self.data_provider.train, batch_size, learning_rate)

      self.log_loss_accuracy(mean_D_loss, acc, epoch, prefix='train_D')

      self.log_loss_accuracy(mean_G_loss, acc, epoch, prefix='train_G')

      summary = tf.Summary(value=[
        tf.Summary.Value(
          tag='learning_rate', simple_value=float(lr_value))
      ])
      self.summary_writer.add_summary(summary, epoch)



      # train_acc=acc
      # train_loss = loss


      # print("Validation...")
      # loss, acc = self.test(
      #   self.data_provider.test, batch_size)
      #
      # self.log_loss_accuracy(loss, acc, epoch, prefix='valid')
      # test_acc=acc
      # test_loss = loss
      #
      # f.write(str(epoch)+','+str(train_acc)+','+str(test_acc))
      # f.write('\n')
      # fx.write(str(epoch) + ',' + str(train_loss) + ',' + str(test_loss))
      # fx.write('\n')

      time_per_epoch = time.time() - start_time
      seconds_left = int((n_epochs - epoch) * time_per_epoch)
      print("Time per epoch: %s, Est. complete in: %s" % (
        str(timedelta(seconds=time_per_epoch)),
        str(timedelta(seconds=seconds_left))))

      self.save_model(global_step=epoch)
      self.summary_writer.add_summary(summary, global_step=epoch)

    total_training_time = time.time() - total_start_time
    # f.close()
    # fx.close()
    print("\n each fold Total training time for all epoches : %s  and %s seconds" % (str(timedelta(
      seconds=total_training_time)),total_training_time))

    fxx = open(self.train_dir + '/timeReport_fold' + str(self.whichFoldData) + '.txt', 'w')
    fxx.write("\n each fold Total training time for all epoches : %s  and %s seconds" % (str(timedelta(
      seconds=total_training_time)),total_training_time))
    fxx.write('\n')
    fxx.close()


  # (Updated)
  def train_one_epoch(self, data, batch_size, learning_rate):
    num_examples = data.num_examples
    total_D_loss = []
    total_G_loss = []
    total_accuracy = []


    z = np.random.normal(0, 1, size=[batch_size, self.n_z]).astype(np.float32)

    for i in range(num_examples // batch_size):
      # videos size is (numpy array):
      #   [batch_size, sequence_length, width, height, channels]
      # labels size is (numpy array):
      #   [batch_size, num_classes]

      d_summary_merge = tf.summary.merge([self.summary_d_loss,
                                          self.summary_d_x_hist,
                                          self.summary_d_z_hist,
                                          self.summary_n_p_x,
                                          self.summary_n_p_z,
                                          self.summary_d_acc])



      mris, labels = data.next_batch(batch_size)

      summary_d, discriminator_loss,lr_value = self.sess.run([d_summary_merge, self.d_loss,self.learning_rate], feed_dict={self.z_vector: z, self.x_vector: mris, self.learning_rate: learning_rate})
      summary_g, generator_loss = self.sess.run([self.summary_g_loss, self.g_loss], feed_dict={self.z_vector: z})

      d_accuracy, n_x, n_z = self.sess.run([self.d_acc, self.n_p_x, self.n_p_z], feed_dict={self.z_vector: z, self.x_vector: mris})
      print n_x, n_z
      if d_accuracy < self.d_thresh:
        self.sess.run([self.optimizer_op_d], feed_dict={self.z_vector: z, self.x_vector: mris})
        print 'Discriminator Training ', "batches_step: ", self.batches_step, ', d_loss:', self.discriminator_loss, 'g_loss:', self.generator_loss, "d_acc: ", self.d_accuracy

      self.sess.run([self.optimizer_op_g], feed_dict={self.z_vector: z})
      print 'Generator Training ', "batches_step: ", self.batches_step, ', d_loss:', self.discriminator_loss, 'g_loss:', self.generator_loss, "d_acc: ", self.d_accuracy

      total_D_loss.append(discriminator_loss)
      total_G_loss.append(generator_loss)
      total_accuracy.append(d_accuracy)
      self.batches_step += 1
      # self.log_loss_accuracy(
      #     loss, accuracy, self.batches_step, prefix='per_batch',
      #     should_print=False)




    mean_D_loss = np.mean(total_D_loss)
    mean_G_loss = np.mean(total_G_loss)
    mean_accuracy = np.mean(total_accuracy)
    return mean_D_loss,mean_G_loss, mean_accuracy,lr_value,summary_d,summary_g

  # (Updated)
  def test(self, data, batch_size):
    num_examples = data.num_examples
    total_loss = []
    total_accuracy = []
    for i in range(num_examples // batch_size):
      batch = data.next_batch(batch_size)
      feed_dict = {
        self.videos: batch[0],
        self.labels: batch[1],
        self.is_training: False,
      }
      fetches = [self.cross_entropy, self.accuracy]
      loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
      total_loss.append(loss)
      total_accuracy.append(accuracy)
    mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)
    return mean_loss, mean_accuracy

  def test_and_record(self, result_file_name, whichFoldData,config,train_dir, data,batch_size):
    evaler = EvalManager(self.train_dir)
    num_examples = data.num_examples
    total_loss = []
    total_accuracy = []
    batch_size=1
    for i in range(num_examples // batch_size):
      batch = data.next_batch(batch_size)
      feed_dict = {
        self.videos: batch[0],
        self.labels: batch[1],
        self.is_training: False,
      }
      fetches = [self.cross_entropy, self.accuracy,self.prediction,self.labels,self.summary_op]
      loss, accuracy,prediction,ground_truth, summary = self.sess.run(fetches, feed_dict=feed_dict)

      total_loss.append(loss)
      total_accuracy.append(accuracy)
      evaler.add_batch_new(prediction, ground_truth)

    mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)
    evaler.report(result_file_name)

    return mean_loss, mean_accuracy

