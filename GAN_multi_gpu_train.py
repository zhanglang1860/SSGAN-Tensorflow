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
from data_providers.utils import get_data_provider_by_path
from models.dense_net_3d import DenseNet3D


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
    dataset_path = os.path.join(r"./datasets/mri/")

    dataset_train, dataset_test, all_hdf5_data = mri.create_default_splits(dataset_path,config.hdf5FileName,config.idFileName,config.cross_validation_number)

    while whichFoldData < config.cross_validation_number:
        data_provider = get_data_provider_by_path(config, dataset_train, dataset_test, all_hdf5_data, whichFoldData)
        model = DenseNet3D(config,data_provider,all_train_dir,whichFoldData)
        if config.train:
            print("Data provider train images: ", data_provider.train.num_examples)
            model.train_all_epochs(config)

        if config.test:
            if not config.train:
                model.load_model()
            print("Data provider test images: ", data_provider.test.num_examples)
            print("Testing...")
            losses = []
            accuracies = []
            # for i in range(10):
            #     loss, accuracy = model.test(data_provider.test, batch_size=config.batch_size)
            #     losses.append(loss)
            #     accuracies.append(accuracy)
            # loss = np.mean(losses)
            # accuracy = np.mean(accuracies)
            # print("mean for repeat run 10times cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
            model.test_and_record(all_result_file_name[whichFoldData], whichFoldData,config,all_train_dir[whichFoldData],data_provider.test, batch_size=config.batch_size)

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
