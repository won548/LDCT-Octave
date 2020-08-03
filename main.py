import argparse
import os
import random

import torch
from torch.backends import cudnn

from test import test
from train import train


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    os.makedirs(args.checkpoints_path, exist_ok=True)
    print('Create path: {}'.format(args.checkpoints_path))

    os.makedirs(args.tensorboard_path, exist_ok=True)
    print('Create path: {}'.format(args.tensorboard_path))

    os.makedirs(args.results_path, exist_ok=True)
    print('Create path: {}'.format(args.results_path))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--model', type=str, default="cnn_oct")
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_patches', type=int, default=64)
    parser.add_argument('--fold', type=str, default="10")

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=500)
    parser.add_argument('--decay_epochs', type=int, default=30)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--valid_epochs', type=int, default=1)
    parser.add_argument('--test_epochs', type=str, default="best")
    parser.add_argument('--test_dates', type=str, default='')
    parser.add_argument('--trunc_min', type=float, default=-1024.0)
    parser.add_argument('--trunc_max', type=float, default=3072.0)

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=6)

    parser.add_argument('--data_path', type=str, default="../../../Datasets/AAPM")
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints')
    parser.add_argument('--tensorboard_path', type=str, default='./tensorboard')
    parser.add_argument('--results_path', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=2020)

    args = parser.parse_args()
    main(args)
