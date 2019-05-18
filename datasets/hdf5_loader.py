from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import h5py
from util import log
from sklearn.model_selection import KFold


class Dataset(object):

    def __init__(self, path, ids, name='default',
                 max_examples=None, is_train=True,hdf5FileName=None):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = hdf5FileName


        file = os.path.join(path, filename)
        log.info("Reading %s ...", file)

        self.data = h5py.File(file, 'r')
        log.info("Reading Done: %s", file)


    def get_data(self, id):
        # preprocessing and data augmentation
        img = self.data[id]['image'].value / 255. * 2 - 1
        l = self.data[id]['label'].value.astype(np.float32)
        return img, l

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(path, hdf5FileName,idFileName,cross_validation_number=10):
    dataset_train, dataset_test = all_ids(path,hdf5FileName,idFileName,cross_validation_number)
    return dataset_train, dataset_test



def all_ids(path,hdf5FileName,idFileName,cross_validation_number):
    id_filename = idFileName
    id_txt = os.path.join(path, id_filename)
    with open(id_txt, 'r') as fp:
        ids = [s.strip() for s in fp.readlines() if s]
    # rs = np.random.RandomState(123)
    # rs.shuffle(ids)
    # create training/testing splits
    # train_ratio = 0.8
    # train_ids = ids[:int(train_ratio*len(ids))]
    # test_ids = ids[int(train_ratio*len(ids)):]
    train_ids =[]
    test_ids = []
    dataset_train = []
    dataset_test = []
    for i in range(cross_validation_number):
        train_ids.append([])
        test_ids.append([])
        dataset_train.append([])
        dataset_test.append([])

    kf = KFold(n_splits=cross_validation_number)
    i = 0


    for cross_train_ids, cross_test_ids in kf.split(ids):
        train_ids_one_fold = []
        test_ids_one_fold = []
        for train_index in range(len(cross_train_ids)):
            train_ids_one_fold.append(ids[cross_train_ids[train_index]])

        for test_index in range(len(cross_test_ids)):
            test_ids_one_fold.append(ids[cross_test_ids[test_index]])


        train_ids[i]=train_ids_one_fold
        test_ids[i]=test_ids_one_fold



        dataset_train[i] = Dataset(path, train_ids[i], name='train', is_train=False, hdf5FileName=hdf5FileName)
        dataset_test[i] = Dataset(path, test_ids[i], name='test', is_train=False, hdf5FileName=hdf5FileName)
        i=i+1

    return dataset_train, dataset_test

