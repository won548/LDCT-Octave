import torch
import torch.nn as nn
from thop import profile

# https://github.com/d-li14/octconv.pytorch
class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(OctConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.low_ch_in, self.high_ch_in = int(self.alpha_in * in_channels), int((1 - self.alpha_in) * in_channels)
        self.low_ch_out, self.high_ch_out = int(self.alpha_out * out_channels), int((1 - self.alpha_out) * out_channels)

        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
            nn.Conv2d(self.low_ch_in, self.low_ch_out, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
            nn.Conv2d(self.low_ch_in, self.high_ch_out, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
            nn.Conv2d(self.high_ch_in, self.low_ch_out, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
            nn.Conv2d(self.high_ch_in, self.high_ch_out, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        if x_h is not None:
            # High -> High conv
            x_h2h = self.conv_h2h(x_h)

            # High -> Low conv
            x_h2l = self.downsample(x_h)
            x_h2l = self.conv_h2l(x_h2l) if self.alpha_out > 0 else None

        if x_l is not None:
            # Low -> High conv
            x_l2h = self.conv_l2h(x_l)
            x_l2h = self.upsample(x_l2h)

            # Low -> Low conv
            x_l2l = self.conv_l2l(x_l) if self.alpha_out > 0 else None

            # Cross Add
            x_h = x_h2h + x_l2h
            x_l = x_l2l + x_h2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l


class OctBlock(nn.Module):
    def __init__(self, ch_in=64, ch_out=64, kernel_size=3, alpha=0.25, mode='middle'):
        super(OctBlock, self).__init__()
        padding = kernel_size // 2
        self.mode = mode
        if self.mode == 'first':        # First
            alpha_in, alpha_out = 0, alpha
        elif self.mode == 'last':       # Last
            alpha_in, alpha_out = alpha, 0
        else:                           # Middle
            alpha_in, alpha_out = alpha, alpha
        self.conv_oct = OctConv(ch_in, ch_out, kernel_size=kernel_size, padding=padding, alpha_in=alpha_in, alpha_out=alpha_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_h, x_l = self.conv_oct(x)
        if type(x) is not tuple:        # First
            x_h, x_l = self.relu(x_h), self.relu(x_l)
        else:
            if self.mode == 'last':     # Last
                x_h = self.relu(x_h)
                return x_h
            else:                       # Middle
                x_h += x[0]             # High
                x_l += x[1]             # Low
                x_h, x_l = self.relu(x_h), self.relu(x_l)
        return x_h, x_l


class CNN_OCT(nn.Module):
    def __init__(self, channels=64, kernel_size=3, alpha=0.25):
        super(CNN_OCT, self).__init__()
        padding = kernel_size // 2

        self.conv_first = nn.Conv2d(1, channels, kernel_size=kernel_size, padding=padding)
        blocks = [OctBlock(ch_in=64, ch_out=64, kernel_size=3, alpha=alpha, mode='first')]
        for i in range(1, 9):
            blocks.append(OctBlock(ch_in=64, ch_out=64, kernel_size=3, alpha=alpha, mode='middle'))
        blocks.append(OctBlock(ch_in=64, ch_out=64, kernel_size=3, alpha=alpha, mode='last'))
        self.blocks = nn.ModuleList(blocks)
        self.conv_recon = nn.Conv2d(channels, 1, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.conv_first(x)
        x = self.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_recon(x)
        x += x_in
        x = self.relu(x)
        return x


if __name__ == "__main__":
    x = torch.ones((16, 1, 64, 64))
    model = CNN_OCT(channels=64, kernel_size=3, alpha=0.75)
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