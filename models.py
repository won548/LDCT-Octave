import torch
import torch.nn as nn

from model.unet import UNet
from model.redcnn import REDCNN
from model.cnn_oct import CNN_OCT


def model_selector(args):
    if args.model == "unet":
        model = UNet(in_ch=1, ch=args.channels, kernel=args.kernel_size)
    if args.model == "redcnn":
        model = REDCNN(out_ch=args.channels, kernel_size=args.kernel_size)
    if args.model == "cnn_oct":
        model = CNN_OCT(channels=args.channels, kernel_size=args.kernel_size, alpha=args.alpha)
    return model


def loss_selector(args):
    if args.loss == 'L1':
        loss = nn.L1Loss()
    elif args.loss == 'L2':
        loss = nn.MSELoss()
    return loss


def init_weights(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.01)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
