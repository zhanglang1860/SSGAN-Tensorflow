import argparse
import os

from model import Model
import datasets.hdf5_loader as dataset


def argparser(is_train=True):
    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MRI',
                        choices=['MNIST', 'SVHN', 'CIFAR10', 'ulna', 'MRI'])
    parser.add_argument('--hdf5FileName', type=str, default='MRIdata_3_AD_MCI_Normal.hdf5',
                        choices=['MRIdata_2_AD_MCI.hdf5', 'MRIdata_2_AD_Normal.hdf5', 'MRIdata_2_MCI_Normal.hdf5',
                                 'MRIdata_3_AD_MCI_Normal.hdf5'])
    parser.add_argument('--idFileName', type=str, default='MRIdata_3_AD_MCI_Normal_id.txt',
                        choices=['MRIdata_2_AD_MCI_id.txt', 'MRIdata_2_AD_Normal_id.txt', 'MRIdata_2_MCI_Normal_id.txt',
                                 'MRIdata_3_AD_MCI_Normal_id.txt'])
    parser.add_argument('--dump_result', type=str2bool, default=False)
    # Model
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_z', type=int, default=128)
    parser.add_argument('--split_dimension_core', type=int, default=10)
    parser.add_argument('--tt_rank', type=int, default=6)
    parser.add_argument('--norm_type', type=str, default='batch',
                        choices=['batch', 'instance', 'None'])
    parser.add_argument('--deconv_type', type=str, default='bilinear',
                        choices=['bilinear', 'nn', 'transpose'])

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=100)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--output_save_step', type=int, default=100)
    # learning
    parser.add_argument('--max_sample', type=int, default=5000,
                        help='num of samples the model can see')
    parser.add_argument('--max_training_steps', type=int, default=60000)
    parser.add_argument('--learning_rate_g', type=float, default=0.000025)
    parser.add_argument('--learning_rate_d', type=float, default=0.00025)
    parser.add_argument('--update_rate', type=int, default=3)
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument('--data_id', nargs='*', default=None)
    # }}}

    config = parser.parse_args()

    dataset_path = os.path.join(r"/data1/wenyu/PycharmProjects/SSGAN-tensor-Tensorflow/datasets",
                                config.dataset.lower())
    dataset_train, dataset_test = dataset.create_default_splits(dataset_path,hdf5FileName=config.hdf5FileName,idFileName=config.idFileName,cross_validation_number=10)
    print("step2")
    img, label = dataset_train[0].get_data(dataset_train[0].ids[0])
    print("step3")
    config.h = img.shape[0]
    config.w = img.shape[1]
    if len(img.shape)==3:
        config.c = img.shape[2]
    else:
        config.c = 1
    config.num_class = label.shape[0]

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)
    return config, model, dataset_train, dataset_test
