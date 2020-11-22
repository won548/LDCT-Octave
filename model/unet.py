import torch
import torch.nn as nn
from thop import profile


class Conv_Block(nn.Module):
    def __init__(self, n_layers=2, in_ch=3, out_ch=64, kernel=3,
                 activation='relu', batch_norm=False, pool=True, residual=True, first_block=True):
        super(Conv_Block, self).__init__()
        self.layers = []
        if activation == 'relu':
            self.act = nn.ReLU()

        if first_block:
            self.layers.append(nn.Conv2d(in_ch, out_ch, kernel, 1, 1))
        else:
            self.layers.append(nn.Conv2d(out_ch, out_ch, kernel, 1, 1))

        if batch_norm:
            self.layers.append(nn.BatchNorm2d(out_ch))
        if activation != 'linear':
            self.layers.append(self.act)

        for layer in range(n_layers - 1):
            self.layers.append(nn.Conv2d(out_ch, out_ch, kernel, 1, 1))
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_ch))
            if activation != 'linear':
                self.layers.append(self.act)
        if pool:
            self.layers.append(nn.MaxPool2d(2))
        self.conv_block = nn.ModuleList(self.layers)
        self.residual = residual

    def forward(self, x):
        for idx, layer in enumerate(self.conv_block):
            if self.residual:
                if idx == len(self.conv_block) - 1:
                    x_res = x
            x = layer(x)
        if self.residual:
            return x, x_res
        else:
            return x


class Upconv_Block(nn.Module):
    def __init__(self, n_layers=2, in_ch=3, out_ch=64, kernel=3,
                 activation='relu', batch_norm=False, residual=True, last_block=False):
        super(Upconv_Block, self).__init__()
        self.layers = []
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'linear':
            self.act = None

        if residual:
            self.layers.append(nn.Conv2d(in_ch, out_ch, kernel, 1, 1))
        else:
            self.layers.append(nn.Conv2d(out_ch, out_ch, kernel, 1, 1))

        if batch_norm:
            self.layers.append(nn.BatchNorm2d(out_ch))
        if activation != 'linear':
            self.layers.append(self.act)

        for layer in range(n_layers - 1):
            self.layers.append(nn.Conv2d(out_ch, out_ch, kernel, 1, 1))
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_ch))
            if activation != 'linear':
                self.layers.append(self.act)

        if last_block is False:
            self.layers.append(nn.Upsample(scale_factor=2))
        self.conv_block = nn.ModuleList(self.layers)

    def forward(self, x):
        for idx, layer in enumerate(self.conv_block):
            x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=3, ch=64, kernel=3, n_depth=3, n_layers=2):
        super(UNet, self).__init__()
        enc_blocks = []
        enc_blocks.append(Conv_Block(n_layers=n_layers, in_ch=in_ch, out_ch=ch, kernel=kernel, first_block=True))

        for depth in range(n_depth-1):
            enc_blocks.append(Conv_Block(n_layers=n_layers, in_ch=ch*(2**depth), out_ch=ch*(2**(depth+1)), kernel=kernel))
        self.encoders = nn.ModuleList(enc_blocks)

        bottom_blocks = []
        bottom_blocks.append(Conv_Block(n_layers=n_layers, in_ch=ch*(2**(depth+1)), out_ch=ch*(2**(depth+2)), kernel=kernel, pool=False, residual=False))
        bottom_blocks.append(nn.Upsample(scale_factor=2))
        self.bottom = nn.ModuleList(bottom_blocks)

        dec_blocks = []
        for depth in reversed(range(n_depth)):
            if depth == 0:
                dec_blocks.append(Upconv_Block(n_layers=n_layers, in_ch=ch*(2**(depth+1))+ch*(2**depth), out_ch=ch*(2**depth), kernel=kernel, residual=True, last_block=True))
            else:
                dec_blocks.append(Upconv_Block(n_layers=n_layers, in_ch=ch*(2**(depth+1))+ch*(2**depth), out_ch=ch*(2**depth), kernel=kernel, residual=True))
        self.decoders = nn.ModuleList(dec_blocks)

        last_blocks = []
        last_blocks.append(Conv_Block(n_layers=1, in_ch=ch, out_ch=ch, kernel=3, pool=False, residual=False))
        last_blocks.append(Conv_Block(n_layers=1, in_ch=ch, out_ch=in_ch, kernel=3, activation='linear', pool=False, residual=False))
        self.last = nn.ModuleList(last_blocks)

    def forward(self, x):
        residuals = []
        residuals.append(x)
        for idx, block in enumerate(self.encoders):
            x, x_res = block(x)
            residuals.append(x_res)

        for idx, block in enumerate(self.bottom):
            x = block(x)

        for idx, block in enumerate(self.decoders):
            x = torch.cat((x, residuals[len(residuals)-idx-1]), dim=1)
            x = block(x)

        for idx, block in enumerate(self.last):
            x = block(x)
        x += residuals[0]
        return x


if __name__ == "__main__":
    # main()
    x = torch.ones((16, 1, 64, 64))
    model = UNet(in_ch=1, ch=48, kernel=3, n_depth=2, n_layers=2)
    z = model(x)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = model.to(device)
    x = x.to(device)
    total_ops, total_params = profile(model, (x,), verbose=False)

    print()
    print("Params(M): {:.4f} ".format(total_params / (1000 ** 2)))
    print(" FLOPs(G): {:.2f}".format(total_ops / (1000 ** 3)))
    print("Output shape:", z.shape)
    print()
