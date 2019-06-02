# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
mri_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by mri_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from config import argparser
from util import log
import mri
import os
import time

from six.moves import xrange
from pprint import pprint
import h5py
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from input_ops import create_input_ops, check_data_id

from util import log
from config import argparser
import datasets.hdf5_loader as dataset
from datetime import timedelta
import random
from operator import itemgetter
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from sklearn.metrics import classification_report



class EvalManager(object):

    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    # def add_batch(self, id, prediction, groundtruth):
    #
    #     # for now, store them all (as a list of minibatch chunks)
    #     self._ids.append(id)
    #     self._predictions.append(prediction)
    #     self._groundtruths.append(groundtruth)

    def compute_accuracy(self, pred, gt):
        correct_prediction = np.sum(np.argmax(pred[:, :-1], axis=1) == np.argmax(gt, axis=1))
        return float(correct_prediction)/pred.shape[0]



    def add_batch_new(self, prediction, groundtruth):

        # for now, store them all (as a list of minibatch chunks)
        shold_be_batch_size=len(prediction)
        for  index in range(shold_be_batch_size):
            self._predictions.append(prediction[index])
            self._groundtruths.append(groundtruth[index])


    def add_batch(self, id, prediction, groundtruth):

        # for now, store them all (as a list of minibatch chunks)
        shold_be_batch_size=len(id)
        for  index in range(shold_be_batch_size):
            self._ids.append(id[index])
            self._predictions.append(prediction[index])
            self._groundtruths.append(groundtruth[index])



    def report(self,result_file_name):
        # report L2 loss
        # log.info("Computing scores...")

        z = zip(self._predictions, self._groundtruths)
        u = list(z)

        with open('./data2/GANresults/' + result_file_name + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['actual_label', 'model_GAN'])
            for pred, gt in u:
                gt_csv = np.argmax(gt)
                pred_csv = np.argmax(pred)

                one_row = []

                one_row.append(str(gt_csv))
                one_row.append(str(pred_csv))

                writer.writerow(one_row)



        # with open('./data2/GANresults/' + all_result_file_name + '.csv', mode='w') as file:
        #     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     if whichFoldData == 0:
        #         writer.writerow(['actual_label', 'model_GAN'])
        #         for pred, gt in u:
        #             gt_csv = np.argmax(gt)
        #             pred_csv = pred[np.argmax(gt)]
        #
        #             one_row = []
        #
        #             one_row.append(str(gt_csv))
        #             one_row.append(str(pred_csv))
        #
        #             writer.writerow(one_row)
        #     else:
        #         for pred, gt in u:
        #             gt_csv = np.argmax(gt)
        #             pred_csv = pred[np.argmax(gt)]
        #
        #             one_row = []
        #
        #             one_row.append(str(gt_csv))
        #             one_row.append(str(pred_csv))
        #
        #             writer.writerow(one_row)







    #
    # def compute_accuracy(self, pred, gt):
    #     correct_prediction = np.sum(np.argmax(pred) == np.argmax(gt))
    #
    #     return float(correct_prediction)/pred.shape[0]

    def report_old(self,result_file_name):
        # report L2 loss
        log.info("Computing scores...")

        score = []
        self._ids = list(map(int, self._ids))

        z = zip(self._ids, self._predictions, self._groundtruths)
        u = list(z)
        v = sorted(u, key=itemgetter(0))

        # using naive method
        # to remove duplicated
        # from list
        v_no_duplicates = []
        for i_original in v:
            if i_original[0] not in [i[0] for i in v_no_duplicates]:
                v_no_duplicates.append(i_original)

        with open('/home/wenyu/Documents/GANresults/'+result_file_name+'.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for id, pred, gt in v_no_duplicates:
                id_csv = id
                pred_csv = pred[np.argmax(gt)]
                gt_csv = np.argmax(gt)
                one_row=[]
                one_row.append(str(id_csv))
                one_row.append(str(gt_csv))
                one_row.append(str(pred_csv))

                writer.writerow(one_row)



        # log.infov("Average accuracy : %.4f", avg*100)




def construct_train_dir(config):
    whichFoldData=0
    all_results_file_name = []
    all_train_dir = []

    temp = config.hdf5FileName.split('.')
    hyper_parameter_all_folds = 'allFolds_{}_lr_g_{}_d_{}_update_G{}D{}_batchSize{}_maxIteration{}'.format(
        temp[0], config.learning_rate_g, config.learning_rate_d,
        1, config.update_rate, config.batch_size, config.max_training_steps
    )

    config.prefix = 'depth{}_growthRate{}_reduce{}_model_type{}_keepPro{}'.format(
        config.depth, config.growth_rate, config.reduction,
        config.model_type, config.keep_prob
    )

    while whichFoldData<config.cross_validation_number:
        hyper_parameter_str = 'fold_{}_{}_lr_g_{}_d_{}_update_G{}D{}_batchSize{}_maxIteration{}'.format(
            str(whichFoldData), temp[0], config.learning_rate_g, config.learning_rate_d,
            1, config.update_rate, config.batch_size, config.max_training_steps
        )

        train_dir = './data2/3dDenseNetTrainDir/train_dir/%s-%s' % (
            config.prefix,
            hyper_parameter_str
        )

        # time.strftime("%Y%m%d-%H%M%S")
        # if tf.gfile.Exists(train_dir):
        #     tf.gfile.DeleteRecursively(train_dir)

        if tf.gfile.Exists(train_dir):
            print()
        else:
            os.makedirs(train_dir)


        log.infov("Train Dir: %s", train_dir)
        result_file_name = hyper_parameter_str + config.prefix + '-' + time.strftime("%Y%m%d-%H%M%S")
        whichFoldData=whichFoldData+1

        all_train_dir.append(train_dir)
        all_results_file_name.append(result_file_name)


    all_result_file_name = hyper_parameter_all_folds + config.prefix
    all_results_file_name.append(all_result_file_name)


    
    return all_train_dir, all_results_file_name
    


def tower_loss(scope, images, labels,config):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.

  logits = mri.inference(images, config)


  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  loss ,accuracy_each_batch,prediction,labels= mri.loss(logits, labels,config)



  # prediction=tf.get_collection('prediction_label', scope)
  # labels=tf.get_collection('target_label', scope)
  #
  # correct_prediction_each_batch = tf.equal(
  #     tf.argmax(prediction, 1),
  #     tf.argmax(labels, 1))
  # accuracy_each_batch2 = tf.reduce_mean(tf.cast(correct_prediction_each_batch, tf.float32))

  tf.summary.scalar("loss/train_accuracy", accuracy_each_batch)
  tf.summary.scalar("loss/D_total_loss_L2", loss)



  return loss, accuracy_each_batch,prediction,labels


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(config,whichFoldData,all_train_dir):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * config.num_gpus.
    total_start_time = time.time()
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images, labels, num_examples_per_epoch_for_train, dataset_test, all_hdf5_data = mri.distorted_inputs(config,whichFoldData)

    num_batches_per_epoch = (num_examples_per_epoch_for_train /
                             config.batch_size / config.num_gpus)
    decay_steps = int(num_batches_per_epoch * mri.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(config.learning_rate_d,
                                    global_step,
                                    decay_steps,
                                    mri.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)








    # Create an optimizer that performs gradient descent.
    opt = tf.train.MomentumOptimizer(
        lr, config.nesterov_momentum, use_nesterov=True)

    # Get images and labels for CIFAR-10.




    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * config.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):

      for i in xrange(config.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (mri.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss, accuracy_each_batch,prediction,labels = tower_loss(scope, image_batch, label_batch,config)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))


    # Group all updates to into a single train op.
    train_op = apply_gradient_op

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement= False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(all_train_dir[whichFoldData], sess.graph)

    for step in xrange(config.max_training_steps):
      start_time = time.time()
      # print("\n", '-' * 30, "Train epoch: %d" % global_step, '-' * 30, '\n')
      print("\n", '-' * 30, "Train epoch: %d" % step, '-' * 30, '\n')
      _, loss_value,accuracy_each_batch_value ,prediction_value,labels_value = sess.run([train_op, loss, accuracy_each_batch,prediction,labels])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = config.batch_size * config.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / config.num_gpus

        format_str = ('%s: step %d, loss = %.2f accuracy_each_batch=%.2f   (%.1f examples/sec; %.3f '
                      'sec/batch)')
        log.infov (format_str % (datetime.now(), step, loss_value,accuracy_each_batch_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % config.output_save_step == 0 or (step + 1) == config.max_training_steps:
        checkpoint = os.path.join(all_train_dir[whichFoldData], 'model.ckpt')
        saver.save(sess, checkpoint, global_step=step)

    total_time = time.time() - total_start_time
    log.infov("total training runtime for one fold all epoches: %s ", total_time)
    return dataset_test, all_hdf5_data


"""Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
def eval_once(saver, summary_writer, predicts,labels, summary_op,train_dir, config,num_test_example,result_file_name, whichFoldData,loss):

    evaler = EvalManager()
    with tf.Session() as sess:

        # if config.checkpoint:
        #
        #     saver.restore(sess, config.checkpoint)
        #     log.info("Loaded from checkpoint!")

      ckpt = tf.train.get_checkpoint_state(train_dir)
      if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

        # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

        batch_size = config.batch_size
        num_iter = int(math.ceil(num_test_example / batch_size))

        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * batch_size


        step=0

        while step < num_iter and not coord.should_stop():
          loss_test, prediction_value, ground_truth = sess.run([loss, predicts, labels])
          tf.add_to_collection('test_losses', loss_test)
          true_count += np.sum(prediction_value)
          step += 1
          evaler.add_batch_new(prediction_value, ground_truth)

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        # print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        total_test_losses = tf.get_collection('test_losses')

        # Calculate the total loss for the current tower.
        total_test_loss = tf.add_n(total_test_losses, name='total_loss')
        tf.summary.scalar("loss/D_test_total_loss", total_test_loss)
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)
      except Exception as e:
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
      evaler.report(result_file_name)
      log.infov("Evaluation complete.")



def eval_run(result_file_name, dataset_test, whichFoldData, all_hdf5_data,config,train_dir):

    log.infov("Start 1-epoch Inference and Evaluation")

    log.info("# of examples = %d", len(dataset_test))

    batch_size= config.batch_size

    with tf.Graph().as_default() as g:

        ids,images, labels = mri.distorted_inputs_test(all_hdf5_data,dataset_test,batch_size, whichFoldData)
        logits = mri.inference(images, config)
        predicts = tf.nn.softmax(logits)
        loss, accuracy_each_batch1,prediction1,labels1= mri.loss(logits, labels, config)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            mri.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(config.eval_dir, g)
        eval_once(saver, summary_writer, predicts,labels, summary_op,train_dir,config,len(dataset_test),result_file_name, whichFoldData,loss)







def calculateConfusionMatrix(each_result_file_name,class_labels):
    df = pd.read_csv('./data2/GANresults/' + each_result_file_name + '.csv')

    df.head()

    cr = classification_report(df.actual_label.values, df.model_GAN.values, target_names=class_labels)
    cm = np.array2string(confusion_matrix(df.actual_label.values, df.model_GAN.values))

    accuracy = accuracy_score(df.actual_label.values, df.predicted_GAN.values)
    # recall = recall_score(df.actual_label.values, df.predicted_GAN.values)
    # precision = precision_score(df.actual_label.values, df.predicted_GAN.values)
    # f1 = f1_score(df.actual_label.values, df.predicted_GAN.values)

    print('Accuracy GAN: %.3f' % (accuracy))
    # print('Recall GAN: %.3f' % (recall))
    # print('Precision GAN: %.3f' % (precision))
    # print('F1 GAN: %.3f' % (f1))

    # fpr_GAN, tpr_GAN, thresholds_GAN = roc_curve(df.actual_label.values, df.model_GAN.values)
    # plt.plot(fpr_GAN, tpr_GAN, 'r-', label='GAN')
    # # plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
    # plt.plot([0, 1], [0, 1], 'k-', label='random')
    # plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    # auc_GAN = roc_auc_score(df.actual_label.values, df.model_GAN.values)

    # print('AUC GAN:%.3f' % auc_GAN)

    # plt.plot(fpr_GAN, tpr_GAN, 'r-', label='GAN AUC: %.3f' % auc_GAN)
    # # plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
    # plt.plot([0, 1], [0, 1], 'k-', label='random')
    # plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()
    # return accuracy, recall, precision, f1, auc_GAN, CM
    return accuracy, cr,cm





def main(argv=None):  # pylint: disable=unused-argument

    whichFoldData = 0


    config = argparser(is_train=True)
    all_train_dir,all_result_file_name = construct_train_dir(config)
    # config.cross_validation_number

    while whichFoldData < config.cross_validation_number:
        if config.train:
            dataset_test, all_hdf5_data=train(config, whichFoldData, all_train_dir)
        if config.test:
            # images, labels, num_examples_per_epoch_for_train, dataset_test, all_hdf5_data = mri.distorted_inputs(config,
            #                                                                                                      whichFoldData)
            eval_run(all_result_file_name[whichFoldData], dataset_test[whichFoldData], whichFoldData, all_hdf5_data,config,all_train_dir[whichFoldData])
        whichFoldData = whichFoldData + 1


    #
    # file_name='fold_4_MRIdata_3_AD_MCI_Normal_lr_g_0.0025_d_0.1_update_G1D6depth40_growthRate12_reduce1.0_batchSize2-20190522-195855'
    # file_name_list=[]
    # file_name_list.append(file_name)
    #
    accuracy_10folds_all = []
    # recall_10folds_all = []
    # precision_10folds_all = []
    # f1_10folds_all = []
    # auc_10folds_all = []
    #
    #

    all_folds_predict_data = []

    with open('./data2/GANresults/' + all_result_file_name[10] + '.csv', "w") as text_file:
        for i in range(10):
            # Open a file: file
            file = open('./data2/GANresults/' + all_result_file_name[i] + '.csv', mode='r')

            # read all lines at once
            all_of_it = file.read()

            if i == 0:
                text_file.write(all_of_it)
            else:
                all_lines = all_of_it.splitlines()
                j = 0
                for each_line in all_lines:
                    if j > 0:
                        text_file.write(each_line)
                        text_file.write('\r\n')
                    j = j + 1

            # close the file
        file.close()

    fold_write = 0
    input_file_name=config.hdf5FileName
    class_labels = []
    name_list = input_file_name.split("_")
    if int(name_list[1]) == 3:
        class_labels.append(name_list[2])
        class_labels.append(name_list[3])
        last_class = name_list[4].split(".")
        class_labels.append(last_class[0])
    else:
        class_labels.append(name_list[2])
        last_class = name_list[3].split(".")
        class_labels.append(last_class[0])

    # for each_result_file_name in trainer.all_results_file_name:
    for each_result_file_name in all_result_file_name:
        # accuracy, recall, precision, f1, auc_GAN , CM = calculateConfusionMatrix(each_result_file_name)
        accuracy, cr,cm = calculateConfusionMatrix(each_result_file_name,class_labels)

        f = open("./GANconfusionMatrixResults/ConfusionMatrix"+str(fold_write)+".txt", 'w')
        log.info("Fold: {}".format(fold_write))
        f.write("Fold: {}".format(fold_write))
        f.write('{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(config.hdf5FileName, cr, cm))
        f.write("accuracy: {}".format(accuracy))
        log.info("accuracy: {}".format(accuracy))
        f.close()
        if fold_write < 10:
            accuracy_10folds_all.append(accuracy)

        # recall_10folds_all.append(recall)
        # precision_10folds_all.append(precision)
        # f1_10folds_all.append(f1)
        # auc_10folds_all.append(auc_GAN)
        fold_write = fold_write+1

    accuracy_10folds_all = np.asarray(accuracy_10folds_all, dtype=np.float32)
    # recall_10folds_all = np.asarray(recall_10folds_all, dtype=np.float32)
    # precision_10folds_all = np.asarray(precision_10folds_all, dtype=np.float32)
    # f1_10folds_all = np.asarray(f1_10folds_all, dtype=np.float32)
    # auc_10folds_all = np.asarray(auc_10folds_all, dtype=np.float32)

    accuracy_10folds_average = np.mean(accuracy_10folds_all)
    # recall_10folds_average = np.mean(recall_10folds_all)
    # precision_10folds_average = np.mean(precision_10folds_all)
    # f1_10folds_average = np.mean(f1_10folds_all)
    # auc_10folds_average = np.mean(auc_10folds_all)


    with open("./GANconfusionMatrixResults/allConfusionMatrix.txt", "w") as text_file:
        text_file.write("Fold: average 10 folds confusion matrix ")
        text_file.write("accuracy: {}".format(accuracy_10folds_average))
        log.info("Fold: average 10 folds confusion matrix ")
        log.info("accuracy: {}".format(accuracy_10folds_average))
        # text_file.write("recall: {}".format(recall_10folds_average))
        # text_file.write("precision: {}".format(precision_10folds_average))
        # text_file.write("f1: {}".format(f1_10folds_average))
        # text_file.write("auc_GAN: {}".format(auc_10folds_average))


    #
    # accuracy_10folds_records, recall_10folds_records, precision_10folds_records, f1_10folds_records, auc_GAN_10folds_records ,CM_GAN_10folds_records= calculateConfusionMatrix(trainer.all_result_file_name)
    #
    # with open("/home/wenyu/Documents/GANresults/ConfusionMatrix.txt", "w") as text_file:
    #     text_file.write("Fold: all 10 folds records ")
    #     text_file.write("accuracy: {}".format(accuracy_10folds_records))
    #     text_file.write("recall: {}".format(recall_10folds_records))
    #     text_file.write("precision: {}".format(precision_10folds_records))
    #     text_file.write("f1: {}".format(f1_10folds_records))
    #     text_file.write("auc_GAN: {}".format(auc_GAN_10folds_records))
    #     text_file.write(np.matrix(CM_GAN_10folds_records))





if __name__ == '__main__':
  tf.app.run()
