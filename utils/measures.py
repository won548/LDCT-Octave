from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import compare_nrmse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from torch.autograd import Variable

from utils.conversion import trunc_denorm


def compute_measure(x, y, pred, data_range=None):
    original_psnr = calculate_psnr(y, x, data_range)
    original_ssim = calculate_ssim(y, x, data_range)
    original_rmse = calculate_nrmse(y, x)
    pred_psnr = calculate_psnr(y, pred, data_range)
    pred_ssim = calculate_ssim(y, pred, data_range)
    pred_rmse = calculate_nrmse(y, pred)
    return (original_psnr, original_ssim, original_rmse), (pred_psnr, pred_ssim, pred_rmse)


def compute_measure_2(x, y, data_range):
    original_psnr = calculate_psnr(x, y, data_range)
    original_ssim = calculate_ssim(x, y, data_range)
    original_rmse = calculate_nrmse(x, y)
    return original_psnr, original_ssim, original_rmse


def interval_mapping(image, cur_min, cur_max, new_min, new_max):
    cur_range = cur_max - cur_min
    new_range = new_max - new_min
    scaled = (image - cur_min) * (new_range / cur_range) + new_min
    return scaled


def calculate_nrmse(img_true, img_test):
    nrmse = compare_nrmse(img_true, img_test)
    return nrmse


def calculate_psnr(img_true, img_test, data_range):
    psnr = compare_psnr(img_true, img_test, data_range)
    return psnr


def calculate_ssim(X, Y, data_range):
    ssim_const = compare_ssim(X, Y,
                              win_size=11,
                              data_range=data_range,
                              gaussian_weights=True,
                              multichannel=False)
    return ssim_const


# Code below only works on torch.tensor
def compute_measure_torch(x, y, y_, device, args):
    # x = trunc_denorm(x.cpu(), args.trunc_max, args.trunc_min)
    # y = trunc_denorm(y.cpu(), args.trunc_max, args.trunc_min)
    # y_ = trunc_denorm(y_.cpu(), args.trunc_max, args.trunc_min)
    x = trunc_denorm(x.detach(), args.trunc_max, args.trunc_min)
    y = trunc_denorm(y.detach(), args.trunc_max, args.trunc_min)
    y_ = trunc_denorm(y_.detach(), args.trunc_max, args.trunc_min)
    data_range = torch.max(y) - torch.min(y)

    original_psnr = compute_PSNR(y, x, data_range)
    original_ssim = compute_SSIM(y, x, data_range, device)
    original_rmse = compute_RMSE(y, x, data_range)
    pred_psnr = compute_PSNR(y, y_, data_range)
    pred_ssim = compute_SSIM(y, y_, data_range, device)
    pred_rmse = compute_RMSE(y, y_, data_range)
    return (original_psnr, original_ssim, original_rmse), (pred_psnr, pred_ssim, pred_rmse)


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


def compute_SSIM(img1, img2, data_range, device="cpu", window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.shape) == 2 or 3:
        shape_ = img1.shape[-1]
        img1 = img1.view(1, 1, shape_, shape_)
        img2 = img2.view(1, 1, shape_, shape_)

    window = create_window(window_size, channel, device)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2).to(device)
    mu2 = F.conv2d(img2, window, padding=window_size//2).to(device)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2).to(device) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2).to(device) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2).to(device) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss_normal = np.array(([exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)]))
    gauss = torch.tensor(gauss_normal, dtype=torch.float32)
    return gauss / gauss.sum()


def create_window(window_size, channel, device="cpu"):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).to(device)
    _2D_window = _1D_window.mm(_1D_window.t()).view(1, 1, window_size, window_size)
    _2D_window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = Variable(_2D_window)
    return window
