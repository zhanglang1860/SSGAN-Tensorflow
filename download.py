from __future__ import print_function
import os
import tarfile
import subprocess
import argparse
import h5py
import numpy as np
import glob
from random import shuffle
import tables
import cv2
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


def prepare_h5py_mri3(image, label, data_dir, num_class=10,shape=None):

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=5000,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'MRIdata_3_AD_MCI_Normal.hdf5'), 'w')
    data_id = open(os.path.join(data_dir, 'MRIdata_3_AD_MCI_Normal_id.txt'), 'w')
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


def download_mnist(download_path):
    data_dir = os.path.join(download_path, 'mnist')

    if check_file(data_dir):
        print('MNIST was downloaded.')
        return

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    # keys = [ 'train-labels-idx1-ubyte.gz']

    # for k in keys:
    #     url = (data_url+k).format(**locals())
    #     target_path = os.path.join(data_dir, k)
    #     print(target_path)
    #     cmd = ['curl', url, '-o', target_path]
    #     print('Downloading ', k)
    #     subprocess.call(cmd)
    #     # cmd = ['gzip', '-d', target_path]
    #     cmd = ['7z', 'e', target_path]
    #     print('Unzip ', k)
    #     # subprocess.call(cmd)
    # print('OK ', k)
    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)).astype(np.float))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)).astype(np.float))

    prepare_h5py(train_image, train_label, test_image, test_label, data_dir)

    # for k in keys:
    #     cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
    #     subprocess.call(cmd)



def convert_to_one_hot(label, num_classes,batch_size):
  """Convert class labels from scalars to one-hot vectors."""
  labels_one_hot = np.zeros((batch_size, num_classes))
  labels_one_hot[np.arange(batch_size), label] = 1

  # labels_one_hot = np.zeros((num_classes,), dtype=np.int)
  # labels_one_hot[label]=1

  return labels_one_hot


def download_ulna(download_path):
    shuffle_data = True
    data_dir = os.path.join(download_path, 'ulna')
    hdf5_path = os.path.join(data_dir, '/data.hdf5')
    all_jpg_path=os.path.join(data_dir, 'all')
    all_jpg_path = os.path.join(all_jpg_path, '/*.jpg')
    all_jpg_path = '/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/ulna/all/*.jpg'

    if check_file(data_dir):
        print('ulna was downloaded.')
        return

    # read addresses and labels from the 'train' folder
    addrs = glob.glob(all_jpg_path)
    labels=[]
    for addr in addrs:
        image_name = addr.split('/')
        temp = image_name[10].split('_')
        labels.append(temp[0])

    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the hata into 60% train, 20% validation, and 20% test
    train_addrs = addrs[0:int(0.6 * len(addrs))]
    train_labels = labels[0:int(0.6 * len(labels))]

    # val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
    # val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]

    test_addrs = addrs[int(0.6 * len(addrs)):]
    test_labels = labels[int(0.6 * len(labels)):]
    train_label = np.array(train_labels, dtype=np.float)
    test_label = np.array(test_labels, dtype=np.float)

    train_images =[]
    test_images = []

    for i in range(len(train_addrs)):
        addr = train_addrs[i]
        img = cv2.imread(addr)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img[..., None]
        train_images.append(gray_img)

    for i in range(len(test_addrs)):
        addr = test_addrs[i]
        img = cv2.imread(addr)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img[..., None]
        test_images.append(gray_img)


    train_image = np.array(train_images, dtype=np.float)
    test_image = np.array(test_images, dtype=np.float)





    prepare_h5py(train_image, train_label, test_image, test_label, data_dir,3)



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


def download_svhn(download_path):
    data_dir = os.path.join(download_path, 'svhn')

    import scipy.io as sio
    # svhn file loader
    def svhn_loader(url, path):
        cmd = ['curl', url, '-o', path]
        subprocess.call(cmd)
        m = sio.loadmat(path)
        return m['X'], m['y']

    if check_file(data_dir):
        print('SVHN was downloaded.')
        return

    data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    train_image, train_label = svhn_loader(data_url, os.path.join(data_dir, 'train_32x32.mat'))

    data_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    test_image, test_label = svhn_loader(data_url, os.path.join(data_dir, 'test_32x32.mat'))

    prepare_h5py(np.transpose(train_image, (3, 0, 1, 2)), train_label,
                 np.transpose(test_image, (3, 0, 1, 2)), test_label, data_dir)

    # cmd = ['rm', '-f', os.path.join(data_dir, '*.mat')]
    # subprocess.call(cmd)


def download_cifar10(download_path):
    data_dir = os.path.join(download_path, 'cifar10')

    # cifar file loader
    def unpickle(file):
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)#dict = cPickle.load(fo, encoding='bytes')
        return dict

    if check_file(data_dir):
        print('CIFAR was downloaded.')
        return

    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    k = 'cifar-10-python.tar.gz'
    target_path = os.path.join(data_dir, k)
    print(target_path)
    #cmd = ['curl', data_url, '-o', target_path]
    print('Downloading CIFAR10')
    #subprocess.call(cmd)
    tarfile.open(target_path, 'r:gz').extractall(data_dir)

    num_cifar_train = 50000
    num_cifar_test = 10000

    target_path = os.path.join(data_dir, 'cifar-10-batches-py')
    train_image = []
    train_label = []
    for i in range(5):
        fd = os.path.join(target_path, 'data_batch_' + str(i + 1))
        dict = unpickle(fd)

        sys.stdout = Logger(r'/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/a.txt')
        print(dict)
        print('------------------')

        train_image.append(dict[b'data'])
        train_label.append(dict[b'labels'])


    train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32 * 32 * 3])
    train_label = np.reshape(np.array(np.stack(train_label, axis=0)), [num_cifar_train])

    fd = os.path.join(target_path, 'test_batch')
    dict = unpickle(fd)
    test_image = np.reshape(dict[b'data'], [num_cifar_test, 32 * 32 * 3])
    test_label = np.reshape(dict[b'labels'], [num_cifar_test])

    prepare_h5py(train_image, train_label, test_image, test_label, data_dir, [32, 32, 3])

    cmd = ['rm', '-f', os.path.join(data_dir, 'cifar-10-python.tar.gz')]
    subprocess.call(cmd)
    cmd = ['rm', '-rf', os.path.join(data_dir, 'cifar-10-batches-py')]
    subprocess.call(cmd)


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
    path = r"/data1/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/all"


    if not os.path.exists(path): os.mkdir(path)

    download_mri2_class(path)
    download_mri3_class(path)

    #
    # if 'MNIST' in args.datasets:
    #     download_mnist(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
    # if 'SVHN' in args.datasets:
    #     download_svhn(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
    # if 'CIFAR10' in args.datasets:
    #     download_cifar10(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
