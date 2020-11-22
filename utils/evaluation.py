from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


# Code below only works on torch.tensor
def compute_measure_torch(x, y, y_):
    x = torch.clamp(x, min=0, max=1.)
    y = torch.clamp(y, min=0, max=1.)
    y_ = torch.clamp(y_, min=0, max=1.)

    original_psnr = compute_PSNR(y, x, 1.)
    original_ssim = compute_SSIM(y, x)
    original_rmse = compute_RMSE(y, x, 1.)
    pred_psnr = compute_PSNR(y, y_, 1.)
    pred_ssim = compute_SSIM(y, y_)
    pred_rmse = compute_RMSE(y, y_, 1.)
    return (original_psnr, original_ssim.item(), original_rmse), (pred_psnr, pred_ssim.item(), pred_rmse)


def compute_PSNR(img1, img2, data_range):
    mse = compute_MSE(img1, img2)
    psnr = 10 * torch.log10(data_range ** 2 / mse)
    return psnr.item()


def compute_MSE(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return mse


def compute_RMSE(img1, img2, data_range):
    mse = compute_MSE(img1, img2)
    rmse = torch.sqrt(mse) / data_range
    return rmse.item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, channel, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def compute_SSIM(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
