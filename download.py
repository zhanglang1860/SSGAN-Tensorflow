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

    # prepare_h5py(train_image, train_label, test_image, test_label, data_dir)
    #
    # # data_order = 'tf'
    # #
    # # img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
    # #
    # # # check the order of data and chose proper data shape to save images
    # # h=96
    # # w=128
    # # c=1
    # # if data_order == 'th':
    # #     data_shape = (0, 3, 224, 224)
    # # elif data_order == 'tf':
    # #     data_shape = (0, h, w, c)
    # #
    # # # open a hdf5 file and create earrays
    # # hdf5_file = tables.open_file(hdf5_path, mode='w')
    # # train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img',
    # #                                         img_dtype, shape=data_shape)
    # # val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img',
    # #                                       img_dtype, shape=data_shape)
    # # test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img',
    # #                                        img_dtype, shape=data_shape)
    # #
    # # mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean',
    # #                                        img_dtype, shape=data_shape)
    # #
    # # # create the label arrays and copy the labels data in them
    # #
    # # hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
    # # hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
    # # hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)
    #
    # # a numpy array to save the mean of the images
    #
    # # mean = np.zeros(data_shape[1:], np.float32)
    # # loop over train addresses
    #
    # for i in range(len(train_addrs)):
    #     if i % 1000 == 0 and i > 1:
    #         print('Train data: {}/{}'.format(i, len(train_addrs)))
    #
    #     addr = train_addrs[i]
    #     img = cv2.imread(addr)
    #     train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32 * 32 * 3])
    #     train_label = np.reshape(np.array(np.stack(train_label, axis=0)), [num_cifar_train])
    #     train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1)).astype(np.float)
    #
    #     fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    #     loaded = np.fromfile(file=fd, dtype=np.uint8)
    #     train_label = np.asarray(loaded[8:].reshape((num_mnist_train)).astype(np.float))
    #     # save the image and calculate the mean so far
    #
    #     # train_storage.append(img[None])
    #     # mean += img / float(len(train_labels))
    #     # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #     #
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # # print how many images are saved every 1000 images
    # # read an image and resize to (224, 224)
    #
    # # cv2 load images as BGR, convert it to RGB
    # # loop over validation addresses
    # # for i in range(len(val_addrs)):
    # #     # print how many images are saved every 1000 images
    # #     if i % 1000 == 0 and i > 1:
    # #         print('Validation data: {}/{}'.format(i, len(val_addrs)))
    # #     # cv2 load images as BGR, convert it to RGB
    # #
    # #     addr = val_addrs[i]
    # #     img = cv2.imread(addr)
    # #     # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    # #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # #     # save the image
    # #     val_storage.append(img[None])
    # #
    # #     # loop over test addresses
    # #
    # #     # add any image pre-processing here
    # #
    # #     # if the data order is Theano, axis orders should change
    # for i in range(len(test_addrs)):
    #     # print how many images are saved every 1000 images
    #
    #     if i % 1000 == 0 and i > 1:
    #         print('Test data: {}/{}'.format(i, len(test_addrs)))
    #
    #     # read an image and resize to (224, 224)
    #     # cv2 load images as BGR, convert it to RGB
    #
    #     addr = test_addrs[i]
    #     img = cv2.imread(addr)
    #     # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     # add any image pre-processing here
    #     # if the data order is Theano, axis orders should change
    #
    #
    #     # save the image
    #     test_storage.append(img[None])
    #     # save the mean and close the hdf5 file
    # mean_storage.append(mean[None])
    # hdf5_file.close()
    #
    # from math import ceil
    # import matplotlib.pyplot as plt
    #
    # num_ulna_train = 450
    # num_ulna_test = 150
    #
    # order = 0
    # for img_name in os.listdir(data_dir):
    #     print
    #     img_name
    #     temp = img_name.split('_')
    #     labels_one_hot=convert_to_one_hot(temp[0], 3)
    #     filename = img_name
    #     fin = open(filename, 'rb')
    #     binary_data = fin.read()
    #
    #     f = h5py.File(os.path.join(data_dir, 'data.hdf5'), 'w')
    #     dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    #     dset = f.create_dataset('binary_data', (100,), dtype=dt)
    #
    #     # Save data string converted as a np array
    #     dset[0] = np.fromstring(binary_data, dtype='uint8')
    #
    #
    #     data_id = open(os.path.join(data_dir, 'id.txt'), 'w')
    #     for i in range(image.shape[0]):
    #
    #         if i % (image.shape[0] / 100) == 0:
    #             bar.update(i / (image.shape[0] / 100))
    #
    #         grp = f.create_group(str(i))
    #         data_id.write(str(i) + '\n')
    #         if shape:
    #             grp['image'] = np.reshape(image[i], shape, order='F')
    #         else:
    #             grp['image'] = image[i]
    #         label_vec = np.zeros(10)
    #         label_vec[label[i] % 10] = 1
    #         grp['label'] = label_vec.astype(np.bool)
    #     bar.finish()
    #     f.close()
    #     data_id.close()
    #     return
    #
    # # open the hdf5 file
    # hdf5_file = tables.open_file(hdf5_path, mode='r')
    #
    # # Total number of samples
    # data_num = hdf5_file.root.train_img.shape[0]
    #
    #
    # fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    # train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1)).astype(np.float)
    #
    # fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    # train_label = np.asarray(loaded[8:].reshape((num_mnist_train)).astype(np.float))
    #
    # fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    # test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1)).astype(np.float)
    #
    # fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    # loaded = np.fromfile(file=fd, dtype=np.uint8)
    # test_label = np.asarray(loaded[8:].reshape((num_mnist_test)).astype(np.float))
    #
    #
    #
    # # for k in keys:
    # #     cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
    # #     subprocess.call(cmd)




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
    path = r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets"
    if not os.path.exists(path): os.mkdir(path)

    download_ulna(path)
    #
    # if 'MNIST' in args.datasets:
    #     download_mnist(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
    # if 'SVHN' in args.datasets:
    #     download_svhn(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
    # if 'CIFAR10' in args.datasets:
    #     download_cifar10(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets")
