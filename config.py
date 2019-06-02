import argparse
import os
import tensorflow as tf
import datasets.hdf5_loader as dataset




def argparser(is_train=True):
    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--eval_dir', type=str, default='./data2/3dDenseNetEvaluate')

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--hdf5FileName', type=str, default='MRIdata_3_AD_MCI_Normal.hdf5',
                        choices=['MRIdata_3_AD_MCI_Normal_test.hdf5','MRIdata_2_AD_MCI.hdf5', 'MRIdata_2_AD_Normal.hdf5', 'MRIdata_2_MCI_Normal.hdf5', 'MRIdata_3_AD_MCI_Normal.hdf5','data.hdf5'])
    parser.add_argument('--idFileName', type=str, default='MRIdata_3_AD_MCI_Normal_id.txt',
                        choices=['MRIdata_3_AD_MCI_Normal_id_test.txt','MRIdata_2_AD_MCI_id.txt', 'MRIdata_2_AD_Normal_id.txt', 'MRIdata_2_MCI_Normal_id.txt',  'MRIdata_3_AD_MCI_Normal_id.txt','id.txt'])
    parser.add_argument('--dump_result', type=str2bool, default=False)
    # Model

    parser.add_argument('--n_z', type=int, default=128)
    parser.add_argument('--cross_validation_number', type=int, default=10)

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ckpt_save_step', type=int, default=50)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--output_save_step', type=int, default=50)
    # learning
    parser.add_argument('--max_sample', type=int, default=50000,
                        help='num of samples the model can see')
    parser.add_argument('--max_training_steps', type=int, default=5)
    parser.add_argument('--reduce_lr_epoch_1', type=int, default=75)
    parser.add_argument('--reduce_lr_epoch_2', type=int, default=110)
    parser.add_argument('--learning_rate_g', type=float, default=0.0025)
    parser.add_argument('--learning_rate_d', type=float, default=0.01)
    parser.add_argument('--update_rate', type=int, default=6)
    parser.add_argument('--num_gpus', type=int, default=2)
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
        '--depth', '-d', type=int, choices=[15, 20, 25, 30, 35, 40, 100, 190, 250],
        default=30,
        help='Depth of whole network, restricted to paper choices')

    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=3, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, default=1.0, metavar='',
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

    return config
