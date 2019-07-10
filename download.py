from __future__ import print_function
import os
import tarfile
import subprocess
import argparse
import h5py
import numpy as np
import glob
from random import shuffle

from math import ceil
import matplotlib.pyplot as plt

import nipy

parser = argparse.ArgumentParser(description='Download dataset for SSGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+', choices=['MNIST', 'SVHN', 'CIFAR10'])


def prepare_h5py(train_image, train_label, test_image, test_label, data_dir, num_class=10,shape=None):
    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hdf5'), 'w')
    data_id = open(os.path.join(data_dir, 'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i % (image.shape[0] / 100) == 0:
            bar.update(i / (image.shape[0] / 100))

        grp = f.create_group(str(i))
        data_id.write(str(i) + '\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(num_class)#10
        index=label[i] % 10
        label_vec[label[i] % 10] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return



def prepare_h5py_mri2_8020(image, label, data_dir, num_class=10,shape=None,first_class_label='0',second_class_label='1', train_test_name='train_'):

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=833,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    hdf5_file_name=train_test_name+'MRIdata_'+ str(num_class) +'_'+first_class_label+'_'+second_class_label+'.hdf5'
    id_file_name=train_test_name+'MRIdata_'+ str(num_class) +'_'+first_class_label+'_'+second_class_label+'_id.txt'

    f = h5py.File(os.path.join(data_dir, hdf5_file_name), 'w')
    data_id = open(os.path.join(data_dir, id_file_name), 'w')
    for i in range(image.shape[0]):

        # if i % (image.shape[0] / 100) == 0:
        #     bar.update(i / (image.shape[0] / 100))

        grp = f.create_group(str(i))
        data_id.write(str(i) + '\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(num_class)#10
        index=label[i] % 10
        label_vec[label[i] % 10] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return


def prepare_h5py_mri2(image, label, data_dir, num_class=10,shape=None,first_class_label='0',second_class_label='1'):

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=833,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    hdf5_file_name='MRIdata_'+ str(num_class) +'_'+first_class_label+'_'+second_class_label+'.hdf5'
    id_file_name='MRIdata_'+ str(num_class) +'_'+first_class_label+'_'+second_class_label+'_id.txt'

    f = h5py.File(os.path.join(data_dir, hdf5_file_name), 'w')
    data_id = open(os.path.join(data_dir, id_file_name), 'w')
    for i in range(image.shape[0]):

        if i % (image.shape[0] / 100) == 0:
            bar.update(i / (image.shape[0] / 100))

        grp = f.create_group(str(i))
        data_id.write(str(i) + '\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(num_class)#10
        index=label[i] % 10
        label_vec[label[i] % 10] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return


def prepare_h5py_mri3(image, label, data_dir, train_test_name, num_class=10,shape=None):

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=5000,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, train_test_name+'MRIdata_3_AD_MCI_Normal.hdf5'), 'w')
    data_id = open(os.path.join(data_dir, train_test_name+'MRIdata_3_AD_MCI_Normal_id.txt'), 'w')
    for i in range(image.shape[0]):

        # if i % (image.shape[0] / 100) == 0:
        #     bar.update(i / (image.shape[0] / 100))

        grp = f.create_group(str(i))
        data_id.write(str(i) + '\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(num_class)#10
        index=label[i] % 10
        label_vec[label[i] % 10] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return




def check_file(data_dir):
    if os.path.exists(data_dir):
        if os.path.isfile(os.path.join('data.hdf5')) and \
                os.path.isfile(os.path.join('id.txt')):
            return True
    else:
        os.mkdir(data_dir)
    return False



def convert_to_one_hot(label, num_classes,batch_size):
  """Convert class labels from scalars to one-hot vectors."""
  labels_one_hot = np.zeros((batch_size, num_classes))
  labels_one_hot[np.arange(batch_size), label] = 1

  # labels_one_hot = np.zeros((num_classes,), dtype=np.int)
  # labels_one_hot[label]=1

  return labels_one_hot



def download_mri2_class(download_path):

    data_dir = os.path.join(download_path, 'mri')

    all_jpg_path = '/data1/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/all/'

    if check_file(data_dir):
        print('mri was downloaded.')
        return



    real_labels = ['AD','MCI','Normal']
    image_class_list = []
    label_class_list = []
    for i in range(len(real_labels)):
        image_class_list.append([])
        label_class_list.append([])

    # for i in range(len(real_labels)):
    #     image_class_list[i] = []
    #     label_class_list[i] = []

    with open("/data1/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/label.csv") as file:
        for line in file:
            strFull = line.split(',')
            strFull[0]=strFull[0]+'.nii'
            each_label = strFull[1].split('\n')
            class_label_index=int(each_label[0])
            image_class_list[class_label_index].append(strFull[0])
            label_class_list[class_label_index].append(each_label[0])


    for i in range(len(real_labels)):
        c = list(zip(image_class_list[i], label_class_list[i]))
        shuffle(c)
        image_class_list[i], label_class_list[i] = zip(*c)
        image_class_list[i]=list(image_class_list[i])
        label_class_list[i] = list(label_class_list[i])

    for i in range(len(real_labels)):
        j = i+1

        while j < len(real_labels):
            addrs=image_class_list[i]+image_class_list[j]


            for ii in range(len(label_class_list[i])):
                label_class_list[i][ii]=0


            for jj in range(len(label_class_list[j])):
                label_class_list[j][jj]=1

            labels = label_class_list[i]+label_class_list[j]
            all_images = []

            for index in range(len(addrs)):
                addr = addrs[index]
                # Get nibabel image object
                mri = nipy.load_image(all_jpg_path + addr).get_data().transpose((1, 0, 2))
                data = np.zeros((109, mri.shape[1], mri.shape[2], 1), dtype='float32')
                mx = mri.max(axis=0).max(axis=0).max(axis=0)
                mri = np.array(mri) / mx
                data[:, :, :, 0] = mri
                mri_final = np.array(data)
                all_images.append(mri_final)

            # def show_img(ori_img):
            #     plt.imshow(ori_img[:, :, 91], cmap='gray')  # channel_last
            #     plt.show()

            all_label = np.array(labels, dtype=np.uint8)
            all_image = np.array(all_images)
            prepare_h5py_mri2(all_image, all_label, data_dir, 2,first_class_label=real_labels[i],second_class_label=real_labels[j])
            j=j+1




def download_mri23_class8020(download_path):

    data_dir = os.path.join(download_path, 'mri')

    all_jpg_path = './datasets/mri/all/'

    if check_file(data_dir):
        print('mri was downloaded.')
        return



    real_labels = ['AD','MCI','Normal']
    image_class_list = []
    label_class_list = []
    for i in range(len(real_labels)):
        image_class_list.append([])
        label_class_list.append([])

    # for i in range(len(real_labels)):
    #     image_class_list[i] = []
    #     label_class_list[i] = []

    with open("./datasets/mri/label.csv") as file:
        for line in file:
            strFull = line.split(',')
            strFull[0]=strFull[0]+'.nii'
            each_label = strFull[1].split('\n')
            class_label_index=int(each_label[0])
            image_class_list[class_label_index].append(strFull[0])
            label_class_list[class_label_index].append(each_label[0])


    for i in range(len(real_labels)):
        c = list(zip(image_class_list[i], label_class_list[i]))
        shuffle(c)
        image_class_list[i], label_class_list[i] = zip(*c)
        image_class_list[i]=list(image_class_list[i])
        label_class_list[i] = list(label_class_list[i])

    all_images_train = []
    all_images_test = []
    all_images_validate = []
    all_labels_train = []
    all_labels_test = []
    all_labels_validate = []

    for i in range(len(real_labels)):
        addrs = image_class_list[i]
        labels = label_class_list[i]

        for index in range(len(addrs)):
            addr = addrs[index]
            # Get nibabel image object
            mri = nipy.load_image(all_jpg_path + addr).get_data().transpose((1, 0, 2))
            data = np.zeros((109, mri.shape[1], mri.shape[2], 1), dtype='float32')
            mx = mri.max(axis=0).max(axis=0).max(axis=0)
            mri = np.array(mri) / mx
            data[:, :, :, 0] = mri
            mri_final = np.array(data)
            if index<27:
                all_images_validate.append(mri_final)
                all_labels_validate.append(labels[index])
            elif index>26 and index<54:
                all_images_test.append(mri_final)
                all_labels_test.append(labels[index])
            else:
                all_images_train.append(mri_final)
                all_labels_train.append(labels[index])

    all_labels_train = np.array(all_labels_train, dtype=np.uint8)
    all_labels_test = np.array(all_labels_test, dtype=np.uint8)
    all_labels_validate = np.array(all_labels_validate, dtype=np.uint8)


    all_images_validate = np.array(all_images_validate)
    all_images_test = np.array(all_images_test)
    all_images_train = np.array(all_images_train)

    prepare_h5py_mri3(all_images_train, all_labels_train, data_dir, "train_", 3)
    prepare_h5py_mri3(all_images_test, all_labels_test, data_dir, "test_", 3)
    prepare_h5py_mri3(all_images_validate, all_labels_validate, data_dir, "validate_", 3)



    for i in range(len(real_labels)):
        j = i+1
        while j < len(real_labels):

            all_images_train = []
            all_images_test = []
            all_images_validate = []
            all_labels_train = []
            all_labels_test = []
            all_labels_validate = []


            addrs=image_class_list[i]+image_class_list[j]

            class_0_count = len(label_class_list[i])
            class_1_count = len(label_class_list[j])


            for ii in range(len(label_class_list[i])):
                label_class_list[i][ii]=0


            for jj in range(len(label_class_list[j])):
                label_class_list[j][jj]=1

            labels = label_class_list[i]+label_class_list[j]

            for index in range(len(addrs)):
                addr = addrs[index]
                # Get nibabel image object
                mri = nipy.load_image(all_jpg_path + addr).get_data().transpose((1, 0, 2))
                data = np.zeros((109, mri.shape[1], mri.shape[2], 1), dtype='float32')
                mx = mri.max(axis=0).max(axis=0).max(axis=0)
                mri = np.array(mri) / mx
                data[:, :, :, 0] = mri
                mri_final = np.array(data)

                if index < int(len(addrs)*0.05):
                    all_images_test.append(mri_final)
                    all_labels_test.append(labels[index])
                elif (index > int(len(addrs)*0.05) and index < 2*int(len(addrs)*0.05)+1):
                    all_images_validate.append(mri_final)
                    all_labels_validate.append(labels[index])
                elif (index > class_0_count and index < class_0_count +int(len(addrs)*0.05)):
                    all_images_test.append(mri_final)
                    all_labels_test.append(labels[index])
                elif (index > class_0_count +int(len(addrs)*0.05)) and (index < class_0_count + 2*(int(len(addrs) * 0.05))):
                    all_images_validate.append(mri_final)
                    all_labels_validate.append(labels[index])
                else:
                    all_images_train.append(mri_final)
                    all_labels_train.append(labels[index])


            # def show_img(ori_img):
            #     plt.imshow(ori_img[:, :, 91], cmap='gray')  # channel_last
            #     plt.show()

            all_labels_train = np.array(all_labels_train, dtype=np.uint8)
            all_images_train = np.array(all_images_train)
            all_images_validate = np.array(all_images_validate)
            all_labels_validate = np.array(all_labels_validate, dtype=np.uint8)

            all_labels_test = np.array(all_labels_test, dtype=np.uint8)
            all_images_test = np.array(all_images_test)

            prepare_h5py_mri2_8020(all_images_train, all_labels_train, data_dir, 2,first_class_label=real_labels[i],second_class_label=real_labels[j],train_test_name='train_')
            prepare_h5py_mri2_8020(all_images_test, all_labels_test, data_dir, 2, first_class_label=real_labels[i],second_class_label=real_labels[j], train_test_name='test_')
            prepare_h5py_mri2_8020(all_images_validate, all_labels_validate, data_dir, 2, first_class_label=real_labels[i], second_class_label=real_labels[j], train_test_name='validate_')

            j=j+1






def download_mri3_class(download_path):
    shuffle_data = True
    data_dir = os.path.join(download_path, 'mri')

    all_jpg_path = '/data1/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/all/'

    if check_file(data_dir):
        print('ulna was downloaded.')
        return


    addrs = []
    labels = []
    with open("/data1/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/label.csv") as file:
        for line in file:
            strFull = line.split(',')
            strFull[0]=strFull[0]+'.nii'
            addrs.append(strFull[0])
            each_label=strFull[1].split('\n')
            labels.append(each_label[0])


    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)



    all_images =[]

    # for i in range(1):
    for i in range(len(addrs)):

        addr = addrs[i]
        # Get nibabel image object
        mri = nipy.load_image(all_jpg_path+addr).get_data().transpose((1,0,2))
        data = np.zeros((109, mri.shape[1], mri.shape[2], 1), dtype='float32')
        mx = mri.max(axis=0).max(axis=0).max(axis=0)
        mri = np.array(mri) / mx
        data[:, :, :, 0] = mri
        mri_final = np.array(data)
        all_images.append(mri_final)

        # def show_img(ori_img):
        #     plt.imshow(ori_img[:, :, 91], cmap='gray')  # channel_last
        #     plt.show()


    all_image = np.array(all_images)
    label = np.array(labels, dtype=np.uint8)


    prepare_h5py_mri3(all_image, label, data_dir,3)

def show_img2(ori_img):
    plt.imshow(ori_img[:, :, 1], cmap='gray')  # channel_last
    plt.show()




import sys
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass




if __name__ == '__main__':
    args = parser.parse_args()
    path = r"./datasets/mri/all"


    if not os.path.exists(path): os.mkdir(path)

    # download_mri2_class(path)
    download_mri23_class8020(path)
    # download_mri3_class(path)

    #
    # if 'MNIST' in args.datasets:
    #     download_mnist(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
    # if 'SVHN' in args.datasets:
    #     download_svhn(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
    # if 'CIFAR10' in args.datasets:
    #     download_cifar10(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
