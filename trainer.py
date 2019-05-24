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

    def __init__(self,config, model):
        self.all_results_file_name = []
        self.step_op = tf.no_op(name='step_no_op')
        self.config = config
        self.model = model

        self.nesterov_momentum = config.nesterov_momentum
        self.weight_decay = config.weight_decay

        print("step2")
        # --- input ops ---
        self.batch_size = config.batch_size

        # --- optimizer ---

        tf.set_random_seed(1234)

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

        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # self.d_optimizer = tf.contrib.layers.optimize_loss(
        #     loss=self.model.d_loss,
        #     global_step=self.global_step,
        #     learning_rate=self.config.learning_rate_d,
        #     optimizer=tf.train.AdamOptimizer(beta1=0.5),
        #     clip_gradients=20.0,
        #     name='d_optimize_loss',
        #     variables=d_var
        # )

        self.d_optimizer = tf.train.MomentumOptimizer(
            self.config.learning_rate_d, self.nesterov_momentum, use_nesterov=True)
        self.train_step = self.d_optimizer.minimize(
            self.model.cross_entropy + l2_loss * self.weight_decay)

        # self.g_optimizer = tf.contrib.layers.optimize_loss(
        #     loss=self.model.g_loss,
        #     global_step=self.global_step,
        #     learning_rate=self.config.learning_rate_g,
        #     optimizer=tf.train.AdamOptimizer(beta1=0.5),
        #     clip_gradients=20.0,
        #     name='g_optimize_loss',
        #     variables=g_var
        # )

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=30000)




        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")



    def train_one_epoch_denseNet(self, data, batch_size, learning_rate):
        _start_time = time.time()
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        total_summary = []
        data._batch_counter = 0

        iteration_time=num_examples // batch_size
        for i in range(iteration_time):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.model.images: images,
                self.model.labels: labels,
                self.model.learning_rate: learning_rate,
                self.model.is_training_denseNet: True,
            }
            fetches = [self.train_step, self.model.cross_entropy, self.model.accuracy,self.summary_op]
            result = self.session.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy,summary = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            # total_summary.append(summary)
            self.batches_step += 1
            self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=True)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        _end_time = time.time()
        return mean_loss, mean_accuracy,(_end_time - _start_time)/int(num_examples // batch_size),summary

    # def train_one_epoch(self, batch, is_train=True):
    #     _start_time = time.time()
    #
    #
    #
    #     batch_chunk = self.session.run(batch)
    #
    #     fetch = [self.global_step, self.train_step,
    #              self.model.cross_entropy, self.model.accuracy,self.summary_op]
    #     #
    #     # if self.model.accuracy < 0.8:
    #     #     fetch.append(self.d_optimizer)
    #     # else:
    #     #     fetch.append(self.g_optimizer)
    #
    #     # if step % (self.config.update_rate+1) > 0:
    #     # Train the generator
    #     # fetch.append(self.d_optimizer)
    #     # else:
    #     # Train the discriminator
    #     #     fetch.append(self.g_optimizer)
    #     fetch.append(self.train_step)
    #
    #
    #     fetch_values = self.session.run(fetch,
    #         feed_dict=self.model.get_feed_dict(batch_chunk)
    #     )
    #
    #     [step, _ ,loss, accuracy, summary] = fetch_values[:5]
    #
    #
    #     _end_time = time.time()
    #
    #
    #
    #     return step, _ , loss,accuracy, summary,\
    #         (_end_time - _start_time)






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



    def train_one_fold_all_epoches(self,dataset_train,train_params, whichFoldData):
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)




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

        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

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
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()


        # pprint(self.batch_train)
        step = self.session.run(self.global_step)

        for epoch in xrange(self.config.max_training_steps):
            self.batches_step = 0

            # periodic inference
            start_time = time.time()
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')

            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                self.config.learning_rate_d = self.config.learning_rate_d / 10
                print("Decrease learning rate, new lr = %f" % self.config.learning_rate_d)

            print("Training...")

            mean_loss, mean_accuracy,step_time,total_summary = \
                self.train_one_epoch_denseNet(dataset_train, batch_size, self.config.learning_rate_d)

            self.log_step_message(epoch, mean_accuracy, mean_loss,
                                       step_time, is_train=False)
            # self.log_loss_accuracy(mean_loss, mean_accuracy, epoch, prefix='train')

            # time_per_epoch = time.time() - start_time
            # seconds_left = int((self.config.max_training_steps - epoch) * time_per_epoch)
            # print("Time per epoch: %s, Est. complete in: %s" % (
            #     str(timedelta(seconds=time_per_epoch)),
            #     str(timedelta(seconds=seconds_left))))


            if epoch % self.config.write_summary_step == 0:
                self.summary_writer.add_summary(total_summary, global_step=step)
              # this checkpoint seems something wrong
            if epoch % self.config.output_save_step == 0:
                log.infov("Saved checkpoint at %d", epoch)
                save_path = self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)

        log.infov("Training ends!")
        _end_time = time.time()
        total_time = _end_time - total_start_time
        log.infov("total training runtime for one fold all epoches: %s ", total_time)




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
        self.dataset = dataset_test[whichFoldData]


        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)


        # max_steps = int(length_dataset / self.batch_size) + 1
        # log.info("max_steps = %d", max_steps)

        evaler = EvalManager()

        num_examples = self.dataset.num_examples
        total_loss = []
        total_accuracy = []
        max_steps = num_examples // self.batch_size
        self.dataset._batch_counter = 0
        for i in range(max_steps):
            batch = self.dataset.next_batch(self.batch_size)
            feed_dict = {
                self.model.images: batch[0],
                self.model.labels: batch[1],
                self.model.is_training_denseNet: False,
            }
            fetches = [self.model.accuracy, self.model.all_preds, self.model.all_targets, self.model.cross_entropy]
            accuracy, prediction_pred, prediction_gt, loss = self.session.run(fetches, feed_dict=feed_dict)
            evaler.add_batch_new(prediction_pred, prediction_gt)


        total_loss.append(loss)
        total_accuracy.append(accuracy)
        evaler.report(result_file_name,all_result_file_name, whichFoldData)

        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)


        log.infov("Fold------"+ str(whichFoldData))
        log.infov("mean accuracy------" + str(mean_accuracy))
        log.infov("Fold------" + str(whichFoldData))


        log.infov("Evaluation complete.")
        return mean_loss, mean_accuracy

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

    # def log_step_message(self, step, accuracy, d_loss, g_loss,
    #                      s_loss, step_time, is_train=True):
    #     if step_time == 0: step_time = 0.001
    #     log_fn = (is_train and log.info or log.infov)
    #     log_fn((" [{split_mode:5s} step {step:4d}] " +
    #             "Supervised loss: {s_loss:.5f} " +
    #             "D loss: {d_loss:.5f} " +
    #             "G loss: {g_loss:.5f} " +
    #             "Accuracy: {accuracy:.5f} "
    #             "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
    #             ).format(split_mode=(is_train and 'train' or 'val'),
    #                      step = step,
    #                      d_loss = d_loss,
    #                      g_loss = g_loss,
    #                      s_loss = s_loss,
    #                      accuracy = accuracy,
    #                      sec_per_batch = step_time,
    #                      instance_per_sec = self.batch_size / step_time
    #                      )
    #            )


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



    print("    FoldData" + str(whichFoldData))

    config, model, dataset_train, dataset_test = argparser(is_train=True)
    trainer = Trainer(config, model)

    log.warning("dataset: %s, learning_rate_g: %f, learning_rate_d: %f",
                config.hdf5FileName, config.learning_rate_g, config.learning_rate_d)




    while whichFoldData < 10:


        if config.train:
            print("Data provider train images: ", len(dataset_train[whichFoldData]))
            trainer.train_one_fold_all_epoches(dataset_train[whichFoldData], train_params_MRI, whichFoldData)
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
