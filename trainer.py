from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from model import Model
from datetime import datetime
from sklearn.metrics import roc_auc_score




train_params_MRI = {
    'batch_size': 3,
    'n_epochs': 10000,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

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



    def report(self,result_file_name, all_result_file_name,whichFoldData):
        # report L2 loss
        # log.info("Computing scores...")

        z = zip(self._predictions, self._groundtruths)
        u = list(z)

        with open('/home/wenyu/Documents/GANresults/' + result_file_name + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['actual_label', 'model_GAN'])
            for pred, gt in u:
                gt_csv = np.argmax(gt)
                pred_csv = pred[np.argmax(gt)]

                one_row = []

                one_row.append(str(gt_csv))
                one_row.append(str(pred_csv))

                writer.writerow(one_row)



        with open('/home/wenyu/Documents/GANresults/' + all_result_file_name + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if whichFoldData == 0:
                writer.writerow(['actual_label', 'model_GAN'])
                for pred, gt in u:
                    gt_csv = np.argmax(gt)
                    pred_csv = pred[np.argmax(gt)]

                    one_row = []

                    one_row.append(str(gt_csv))
                    one_row.append(str(pred_csv))

                    writer.writerow(one_row)
            else:
                for pred, gt in u:
                    gt_csv = np.argmax(gt)
                    pred_csv = pred[np.argmax(gt)]

                    one_row = []

                    one_row.append(str(gt_csv))
                    one_row.append(str(pred_csv))

                    writer.writerow(one_row)







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





class Trainer(object):

    def __init__(self,config):
        self.all_results_file_name = []

        self.config = config


        self.nesterov_momentum = config.nesterov_momentum
        self.weight_decay = config.weight_decay

        print("step2")
        # --- input ops ---
        self.batch_size = config.batch_size

        # --- optimizer ---


        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()

        d_var = [v for v in all_var if v.name.startswith('Discriminator')]
        log.warn("********* d_var ********** ")
        slim.model_analyzer.analyze_vars(d_var, print_info=True)

        # g_var = [v for v in all_var if v.name.startswith(('Generator'))]
        # log.warn("********* g_var ********** ")
        # slim.model_analyzer.analyze_vars(g_var, print_info=True)
        # rem_var = (set(all_var) - set(d_var) - set(g_var))

        rem_var = (set(all_var) - set(d_var))
        print([v.name for v in rem_var])
        assert not rem_var

        # self.ckpt_path = config.checkpoint
        # if self.ckpt_path is not None:
        #     log.info("Checkpoint path: %s", self.ckpt_path)
        #     self.saver.restore(self.session, self.ckpt_path)
        #     log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def tower_loss(self, scope, images, labels):
        """Calculate the total loss on a single tower running the CIFAR model.

          Args:
            scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
            images: Images. 4D tensor of shape [batch_size, height, width, 3].
            labels: Labels. 1D tensor of shape [batch_size].

          Returns:
             Tensor of shape [] containing the total loss for a batch of data
          """

        # Build inference Graph.
        # --- create model ---
        with tf.device('/cpu:0'):
            is_train=True
            model = Model(self.config, self.config.growth_rate, self.config.depth,
                          self.config.total_blocks, self.config.keep_prob,
                          self.config.nesterov_momentum, self.config.model_type, scope, debug_information=self.config.debug,
                          is_train=is_train,
                          reduction=self.config.reduction,
                          bc_mode=self.config.bc_mode)



            loss, accuracy_each_batch, accuracy_all_batch, prediction_all, labels_all= model.build(images,labels, self.config.weight_decay,is_train=True)

        return loss, accuracy_each_batch, accuracy_all_batch, prediction_all, labels_all

    def average_gradients(self,tower_grads):
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

    def train_all_epoch_denseNet(self, data, batch_size, learning_rate):
        _start_time = time.time()
        num_gpus = 2
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * FLAGS.num_gpus.
            total_start_time = time.time()
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Calculate the learning rate schedule.
            num_batches_per_epoch = (len(data.ids) /batch_size/ num_gpus)
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(self.config.learning_rate_d,
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            d_optimizer = tf.train.MomentumOptimizer(
                lr, self.nesterov_momentum, use_nesterov=True)

            _, batch_train = create_input_ops(
                data, batch_size)
            images, labels = batch_train['image'], batch_train['label']

            batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * num_gpus)

            # Calculate the gradients for each model tower.
            tower_grads = []

            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('MRI_gpu_%d' % (i)) as scope:
                            # Dequeues one batch for the GPU
                            image_batch, label_batch = batch_queue.dequeue()
                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss, accuracy_each_batch, accuracy_all_batch, prediction_all, labels_all = self.tower_loss(scope, image_batch, label_batch)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = d_optimizer.compute_gradients(loss)
                            print("qqqqqqqqqqqqqq")
                            print(grads)
                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            print("ZZZZZZZZZZZZZZZZZZZZZZ")
            print(tower_grads[0])
            grads = self.average_gradients(tower_grads)
            print(grads)

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = d_optimizer.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                 MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge(summaries)

            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))
            self.sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=self.sess)

            summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)

            print("XXXXXXXXXXXXXXXXXXXXXX")

            print("self.config.max_training_steps:%s",self.config.max_training_steps)


            for epoch in xrange(self.config.max_training_steps):
                start_time = time.time()

                print("self.epoch:%s", epoch)
                print("XXXXXX~~~~~~~~~~~~~~~~~~~XXXXXXXXX")
                print(grads)
                print(loss)

                _, loss_value = self.sess.run([train_op, loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if epoch % 10 == 0:
                    num_examples_per_epoch = batch_size * num_gpus
                    examples_per_sec = num_examples_per_epoch / duration
                    sec_per_batch = duration / num_gpus

                    format_str = ('%s: epoch %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    log.info(format_str % (datetime.now(), epoch, loss_value,
                                        examples_per_sec, sec_per_batch))

                if epoch % 100 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, epoch)


                # Save the model checkpoint periodically.
                if epoch % 50 == 0 or (epoch + 1) == self.config.max_training_steps:
                    log.infov("Saved checkpoint at %d", epoch)
                    checkpoint_path = os.path.join(self.train_dir, 'model')
                    saver.save(self.sess, checkpoint_path, global_step=epoch)

            log.infov("Training ends!")
            _end_time = time.time()

            total_time = _end_time - total_start_time
            log.infov("total training runtime for one fold all epoches: %s ", total_time)



    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)



    def train_one_fold_all_epoches(self,dataset_train, whichFoldData):


        temp = self.config.hdf5FileName.split('.')
        hyper_parameter_str = 'fold_{}_{}_lr_g_{}_d_{}_update_G{}D{}'.format(
            str(whichFoldData), temp[0], self.config.learning_rate_g, self.config.learning_rate_d,
            1, self.config.update_rate
        )

        hyper_parameter_all_folds = 'allFolds_{}_lr_g_{}_d_{}_update_G{}D{}'.format(
            temp[0], self.config.learning_rate_g, self.config.learning_rate_d,
            1, self.config.update_rate
        )

        self.config.prefix = 'depth{}_growthRate{}_reduce{}_batchSize{}'.format(
            self.config.depth, self.config.growth_rate, self.config.reduction,
            self.config.batch_size
        )

        self.train_dir = './train_dir/%s-%s' % (
            self.config.prefix,
            hyper_parameter_str
        )

        # time.strftime("%Y%m%d-%H%M%S")
        if tf.gfile.Exists(self.train_dir):
            tf.gfile.DeleteRecursively(self.train_dir)

        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)



        self.checkpoint = self.config.checkpoint
        if self.checkpoint is None and self.train_dir:
            self.checkpoint = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            # self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint)



        self.result_file_name = hyper_parameter_str + self.config.prefix + '-'+time.strftime("%Y%m%d-%H%M%S")
        self.all_result_file_name = hyper_parameter_all_folds + self.config.prefix

        self.all_results_file_name.append(self.result_file_name)






        log.infov("Training Starts!")
        pprint(dataset_train)
        batch_size = self.config.batch_size



        self.train_all_epoch_denseNet(dataset_train, batch_size, self.config.learning_rate_d)






    def log_step_message(self, step, accuracy, d_loss,
                          step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "D loss: {d_loss:.5f} " +
                "Accuracy: {accuracy:.5f} "
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step = step,
                         d_loss = d_loss,
                         accuracy = accuracy,
                         sec_per_batch = step_time,
                         instance_per_sec = self.batch_size / step_time
                         )
               )


    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def eval_run_old(self,result_file_name,dataset_test, whichFoldData):
        # load checkpoint

        self.dataset = dataset_test[whichFoldData]

        ###############################################33
        check_data_id(dataset_test[whichFoldData], self.config.data_id)
        scope_name = 'Fold_Data_'+ str(whichFoldData)

        _, self.batch = create_input_ops(dataset_test[whichFoldData], self.batch_size,
                                         data_id=self.config.data_id,
                                         is_training=False,scope=scope_name,
                                         shuffle=False)

        ############# here for 10 cross validation   ###################
        ##############################################################3



        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)


        max_steps = int(length_dataset / self.batch_size) + 1
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager()
        try:
            for s in xrange(max_steps):
                step, accuracy, step_time, batch_chunk, prediction_pred, prediction_gt , loss= \
                    self.run_single_step(self.batch)
                self.log_step_message(s,accuracy, loss, step_time,is_train = False)
                evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        evaler.report(result_file_name)
        log.infov("Evaluation complete.")


    def eval_run(self,result_file_name,all_result_file_name, dataset_test, whichFoldData):
        dataset = dataset_test[whichFoldData]


        if self.checkpoint:
            self.saver.restore(self.sess, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        _start_time = time.time()
        num_gpus = 2
        with tf.Graph().as_default(), tf.device('/cpu:0'):

            _, batch_test = create_input_ops(
                dataset, self.batch_size)
            images, labels = batch_test['image'], batch_test['label']
            batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * num_gpus)

            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('MRI_gpu_test_%d' % (i)) as scope:
                            # Dequeues one batch for the GPU
                            image_batch, label_batch = batch_queue.dequeue()
                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss, accuracy_each_batch, accuracy_all_batch, prediction_all, labels_all = self.tower_loss(scope,
                                                                                                      image_batch,
                                                                                                      label_batch)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            evaler = EvalManager()


            evaler.add_batch_new(prediction_all, labels_all)


            evaler.report(result_file_name,all_result_file_name, whichFoldData)


    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, accuracy, all_preds, all_targets, _, loss] = self.session.run(
            [self.global_step, self.model.accuracy, self.model.all_preds, self.model.all_targets, self.step_op,self.model.cross_entropy],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, accuracy, (_end_time - _start_time), batch_chunk, all_preds, all_targets, loss

    def log_step_message_test(self, step, accuracy, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "batch total-accuracy (test): {test_accuracy:.2f}% " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         test_accuracy=accuracy*100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
                         )
               )




def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return # your code here
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return # your code here
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return # your code here


def find_conf_matrix_values(y_true,y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true,y_pred)
    FN = find_FN(y_true,y_pred)
    FP = find_FP(y_true,y_pred)
    TN = find_TN(y_true,y_pred)
    return TP,FN,FP,TN


def my_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])




def calculateConfusionMatrix(each_result_file_name):
    df = pd.read_csv('/home/wenyu/Documents/GANresults/' + each_result_file_name + '.csv')
    df.head()
    thresh = 0.5
    df['predicted_GAN'] = (df.model_GAN >= thresh).astype('int')
    df.head()

    CM = confusion_matrix(df.actual_label.values, df.predicted_GAN.values)



    accuracy = accuracy_score(df.actual_label.values, df.predicted_GAN.values)
    # recall = recall_score(df.actual_label.values, df.predicted_GAN.values)
    # precision = precision_score(df.actual_label.values, df.predicted_GAN.values)
    # f1 = f1_score(df.actual_label.values, df.predicted_GAN.values)

    print('scores with threshold = 0.5')
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
    return accuracy, CM





def main():


    whichFoldData = 0



    print("xxxx    FoldData" + str(whichFoldData))

    config= argparser(is_train=True)

    trainer = Trainer(config)

    log.warning("dataset: %s, learning_rate_g: %f, learning_rate_d: %f",
                config.hdf5FileName, config.learning_rate_g, config.learning_rate_d)




    while whichFoldData < 1:


        if config.train:
            print("Data provider train images: ", len(dataset_train[whichFoldData]))
            trainer.train_one_fold_all_epoches(dataset_train[whichFoldData], whichFoldData)
        if config.test:
            print("Data provider test images: ", len(dataset_test[whichFoldData]))
            print("Testing...")
            trainer.eval_run(trainer.result_file_name,trainer.all_result_file_name, dataset_test,whichFoldData)
        whichFoldData=whichFoldData+1




    file_name='fold_4_MRIdata_3_AD_MCI_Normal_lr_g_0.0025_d_0.1_update_G1D6depth40_growthRate12_reduce1.0_batchSize2-20190522-195855'
    file_name_list=[]
    file_name_list.append(file_name)

    accuracy_10folds_all = []
    recall_10folds_all = []
    precision_10folds_all = []
    f1_10folds_all = []
    auc_10folds_all = []


    fold_write = 0



    # for each_result_file_name in trainer.all_results_file_name:
    for each_result_file_name in file_name_list:
        # accuracy, recall, precision, f1, auc_GAN , CM = calculateConfusionMatrix(each_result_file_name)
        accuracy, CM = calculateConfusionMatrix(each_result_file_name)

        with open("/home/wenyu/Documents/GANresults/ConfusionMatrix.txt", "w") as text_file:
            text_file.write("Fold: {}".format(fold_write))
            text_file.write("accuracy: {}".format(accuracy))
            # text_file.write("recall: {}".format(recall))
            # text_file.write("precision: {}".format(precision))
            # text_file.write("f1: {}".format(f1))
            # text_file.write("auc_GAN: {}".format(auc_GAN))
            text_file.write("Confusion matrix for fold {}".format(fold_write))

            text_file.write(np.matrix(CM))

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
    auc_10folds_all = np.asarray(auc_10folds_all, dtype=np.float32)

    accuracy_10folds_average = np.mean(accuracy_10folds_all)
    # recall_10folds_average = np.mean(recall_10folds_all)
    # precision_10folds_average = np.mean(precision_10folds_all)
    # f1_10folds_average = np.mean(f1_10folds_all)
    auc_10folds_average = np.mean(auc_10folds_all)


    with open("/home/wenyu/Documents/GANresults/ConfusionMatrix.txt", "w") as text_file:
        text_file.write("Fold: average 10 folds confusion matrix ")
        text_file.write("accuracy: {}".format(accuracy_10folds_average))
        # text_file.write("recall: {}".format(recall_10folds_average))
        # text_file.write("precision: {}".format(precision_10folds_average))
        # text_file.write("f1: {}".format(f1_10folds_average))
        text_file.write("auc_GAN: {}".format(auc_10folds_average))


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
    main()
