import torch
import torch.nn as nn

from model.cnn_oct import CNN_OCT


def model_selector(args):
    if args.model == "cnn_oct":
        model = CNN_OCT(channels=args.channels, kernel_size=args.kernel_size, alpha=args.alpha)
    model.apply(init_weights)
    return model


def loss_selector():
    loss = nn.L1Loss()
    return loss


def init_weights(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.01)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
