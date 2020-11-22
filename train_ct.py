import random
from argparse import ArgumentParser

import torch
from trainer import Trainer
from torch.backends import cudnn
from datasets import load_CTDataset

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('--base_dir', type=str, default='./runs')
    parser.add_argument('--data_dir', help='Your dataset path', default='/home/dongkyu/Datasets/AAPM/')
    parser.add_argument('--test_fold', help='dataset path', type=str, default="10")

    # Model parameters
    parser.add_argument('--model', help='model', default='redcnn', type=str)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--alpha', help='Alpha value for Octave convolution', type=float, default=0.25)

    # Training hyperparameters
    parser.add_argument('--learning_rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('--batch_size', help='minibatch size', default=64, type=int)
    parser.add_argument('--num_patches', help='Numbers of patches in one image', type=int, default=64)
    parser.add_argument('--num_epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('--val_epoch', help='validation epoch', default=1, type=int)
    parser.add_argument('--loss', help='loss function', choices=['L1', 'L2'], default='L2', type=str)
    parser.add_argument('--patch_size', help='random crop size', default=64, type=int)
    parser.add_argument('--scheduler', help='scheduler on/off', action='store_true')

    # Device parameters
    parser.add_argument('--device', help='GPU for train', default='0', type=str)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    """ Train LDCT Denoising using Octave convolution """
    # Parse training parameters
    params = parse_args()

    # Seed settings
    # torch.manual_seed(params.seed)
    # torch.cuda.manual_seed(params.seed)
    # torch.cuda.manual_seed_all(params.seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(params.seed)
    # random.seed(params.seed)

    if params.model != 'cnn_oct':
        params.alpha = None

    # Train/valid datasets
    dataloader_train = load_CTDataset(data_dir=params.data_dir, params=params, mode='train')
    dataloader_valid = load_CTDataset(data_dir=params.data_dir, params=params, mode='valid')

    # Initialize model and train
    train = Trainer(params=params, mode='train')
    train.train(dataloader_train, dataloader_valid)
