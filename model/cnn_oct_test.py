import torch
import torch.nn as nn
from thop import profile


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


class CNN_OCT_TEST(nn.Module):
    def __init__(self, channels=64, kernel_size=3, alpha=0.25):
        super(CNN_OCT_TEST, self).__init__()
        if kernel_size == 3:
            self.padding = 1
        elif kernel_size == 5:
            self.padding = 2

        self.channels = channels
        self.kernel_size = kernel_size
        self.alpha = alpha

        self.conv_first = nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, padding=self.padding)
        self.conv_oct_0 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=0, alpha_out=self.alpha)
        self.conv_oct_1 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_2 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_3 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_4 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_5 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_6 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_7 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_8 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=self.alpha)
        self.conv_oct_9 = OctConv(self.channels, self.channels, kernel_size=self.kernel_size, padding=self.padding, alpha_in=self.alpha, alpha_out=0)
        self.recon = nn.Conv2d(self.channels, 1, kernel_size=self.kernel_size, padding=self.padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_res = x
        x = self.conv_first(x)
        x_res_in = x

        x_h, x_l = self.conv_oct_0(x)
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res0, x_l_res0 = x_h, x_l

        x_h, x_l = self.conv_oct_1((x_h, x_l))
        x_h += x_h_res0
        x_l += x_l_res0
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res1, x_l_res1 = x_h, x_l

        x_h, x_l = self.conv_oct_2((x_h, x_l))
        x_h += x_h_res1
        x_l += x_l_res1
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res2, x_l_res2 = x_h, x_l

        x_h, x_l = self.conv_oct_3((x_h, x_l))
        x_h += x_h_res2
        x_l += x_l_res2
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res3, x_l_res3 = x_h, x_l

        x_h, x_l = self.conv_oct_4((x_h, x_l))
        x_h += x_h_res3
        x_l += x_l_res3
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res4, x_l_res4 = x_h, x_l

        x_h, x_l = self.conv_oct_5((x_h, x_l))
        x_h += x_h_res4
        x_l += x_l_res4
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res5, x_l_res5 = x_h, x_l

        x_h, x_l = self.conv_oct_6((x_h, x_l))
        x_h += x_h_res5
        x_l += x_l_res5
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res6, x_l_res6 = x_h, x_l

        x_h, x_l = self.conv_oct_7((x_h, x_l))
        x_h += x_h_res6
        x_l += x_l_res6
        x_h, x_l = self.relu(x_h), self.relu(x_l)
        x_h_res7, x_l_res7 = x_h, x_l

        x_h, x_l = self.conv_oct_8((x_h, x_l))
        x_h += x_h_res7
        x_l += x_l_res7
        x_h, x_l = self.relu(x_h), self.relu(x_l)

        x_h, x_l = self.conv_oct_9((x_h, x_l))
        x_h += x_res_in
        x = self.relu(x_h)

        x = self.recon(x)
        x += x_res

        return x


def main():
    x = torch.ones((16, 1, 64, 64))
    model = CNN_OCT_TEST(channels=64, kernel_size=5, alpha=0.5)
    # z = model(x)
    params = 0
    for name, p in model.named_parameters():
        p_sum = 1
        # print(name, "\t", p.shape)
        for p_s in p.shape:
            p_sum *= p_s
        params += p_sum

    print(params)
    print("Params(M) | FLOPs(G)")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = model.to(device)
    x = x.to(device)
    total_ops, total_params = profile(model, (x,), verbose=True)
    print("%f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))


if __name__ == "__main__":
    main()
