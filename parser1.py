from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV TA\'s tutorial in image classification using pytorch')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='./hw3_data/',
                        help="root path to data directory")

    parser.add_argument('--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=100, type=int,
                        help="num of validation iterations")
    parser.add_argument('--val_epoch', default=16, type=int,
                        help="num of validation iterations")
    parser.add_argument('--train_batch', default=32,type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int,
                        help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help="initial learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument('--img_path', default="./hw2_data/val/img", type=str)
    # parser.add_argument('--seg_path', default="/Users/celine/Desktop/DLCV/Ex2/hw2-huetufemchopf/outputfiles", type=str)

    # resume trained model
    parser.add_argument('--resume', type=str, default='./log/model_bestbest.pth.tar',
                        help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--sample_dir', type=str, default='sample')
    parser.add_argument('--acgan_dir', type=str, default='acgan')

    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args