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
    parser.add_argument('--hdf5FileName', type=str, default='MRIdata_3_AD_MCI_Normal.hdf5',
                        choices=['MRIdata_2_AD_MCI.hdf5', 'MRIdata_2_AD_Normal.hdf5', 'MRIdata_2_MCI_Normal.hdf5', 'MRIdata_3_AD_MCI_Normal.hdf5'])
    parser.add_argument('--idFileName', type=str, default='MRIdata_3_AD_MCI_Normal_id.txt',
                        choices=['MRIdata_2_AD_MCI_id.txt', 'MRIdata_2_AD_Normal_id.txt', 'MRIdata_2_MCI_Normal_id.txt',  'MRIdata_3_AD_MCI_Normal_id.txt'])
    parser.add_argument('--dump_result', type=str2bool, default=False)
    # Model

    parser.add_argument('--n_z', type=int, default=128)
    parser.add_argument('--cross_validation_number', type=int, default=10)

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--ckpt_save_step', type=int, default=50)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--output_save_step', type=int, default=100)
    # learning
    parser.add_argument('--max_sample', type=int, default=5000,
                        help='num of samples the model can see')
    parser.add_argument('--max_training_steps', type=int, default=1)
    parser.add_argument('--learning_rate_g', type=float, default=0.0025)
    parser.add_argument('--learning_rate_d', type=float, default=0.1)
    parser.add_argument('--update_rate', type=int, default=6)
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument('--data_id', nargs='*', default=None)
    # }}}

    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='What type of model to use')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 40],
        default=12,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[40, 100, 190, 250],
        default=40,
        help='Depth of whole network, restricted to paper choices')

    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=3, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        help="Keep probability for dropout.")

    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')






    config = parser.parse_args()



    if not config.keep_prob:
        config.keep_prob = 1.0


    if config.model_type == 'DenseNet':
        config.bc_mode = False
        config.reduction = 1.0
    elif config.model_type == 'DenseNet-BC':
        config.bc_mode = True

    model_params = vars(config)

    if not config.train and not config.test:
        print("You should train or test your network. Please check params.")
        exit()



    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))


    print("Prepare training data...")




    dataset_path = os.path.join(r"/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-original-Tensorflow/datasets/mri/")

    dataset_train, dataset_test = dataset.create_default_splits(dataset_path,hdf5FileName=config.hdf5FileName,idFileName=config.idFileName,cross_validation_number=config.cross_validation_number)
    #dataset_train, dataset_test are 10 cross validation data.
    #dataset_train[i] is the i-th fold data
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
    model = Model(config, config.growth_rate, config.depth,
    config.total_blocks, config.keep_prob,
    config.nesterov_momentum, config.model_type, debug_information=config.debug, is_train=is_train,
    reduction = config.reduction,
    bc_mode = config.bc_mode)
    return config, model, dataset_train, dataset_test
