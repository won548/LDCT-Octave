# from datasets import load_dataset
from argparse import ArgumentParser

from datasets import load_CTDataset
from trainer import Trainer


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('--base_dir', type=str, default='./runs')
    parser.add_argument('--data_dir', help='dataset path', default='/home/dongkyu/Datasets/AAPM/')
    parser.add_argument('--test_fold', help='dataset path', type=str, default="10")
    parser.add_argument('--subject', help='subject', default='L506')
    parser.add_argument('--desc', help='description', default='test')

    # Model parameters
    parser.add_argument('--model', help='model', default='redcnn', type=str)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--alpha', help='Alpha value for Octave convolution', type=float, default=0.25)

    # Testing parameters
    parser.add_argument('--test_epoch', type=str, default='best')
    parser.add_argument('--test_date', type=str, default='')
    parser.add_argument('--patch_size', help='random crop size', default=64, type=int)
    parser.add_argument('--device', help='GPU for test', default='0', type=str)

    # Device parameters
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    """ Test Noise2Noise. """
    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    dataloader_test = load_CTDataset(data_dir=params.data_dir, params=params, mode='test')

    # Initialize model and train
    train = Trainer(params=params, mode='test')
    train.test(dataloader_test)
