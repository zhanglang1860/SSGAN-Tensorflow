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
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'SVHN', 'CIFAR10','ulna'])
    parser.add_argument('--dump_result', type=str2bool, default=False)
    # Model
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_z', type=int, default=128)
    parser.add_argument('--norm_type', type=str, default='batch',
                        choices=['batch', 'instance', 'None'])
    parser.add_argument('--deconv_type', type=str, default='bilinear',
                        choices=['bilinear', 'nn', 'transpose'])

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=10000)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--output_save_step', type=int, default=1000)
    # learning
    parser.add_argument('--max_sample', type=int, default=5000,
                        help='num of samples the model can see')
    parser.add_argument('--max_training_steps', type=int, default=10000)
    parser.add_argument('--learning_rate_g', type=float, default=5e-3)
    parser.add_argument('--learning_rate_d', type=float, default=5e-3)
    parser.add_argument('--update_rate', type=int, default=1)
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument('--data_id', nargs='*', default=None)
    # }}}

    config = parser.parse_args()

    dataset_path = os.path.join(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets",
                                config.dataset.lower())
    dataset_train, dataset_test = dataset.create_default_splits(dataset_path)
    print("step2")
    img, label = dataset_train.get_data(dataset_train.ids[0])

    print("step3")
    config.h = img.shape[0]
    config.w = img.shape[1]

    if img.shape==3:
        config.c = img.shape[2]
    else:
        config.c = 1


    config.num_class = label.shape[0]

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)
    return config, model, dataset_train, dataset_test
