import glob
import os

import SimpleITK as sitk
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
import torch


def plot_save_img(input_img, pred_img, target_img, save_cfg):
    input_img = input_img.squeeze()
    pred_img = pred_img.squeeze()
    target_img = target_img.squeeze()

    input_img = torch.clamp(input_img, min=0, max=1.)
    pred_img = torch.clamp(pred_img, min=0, max=1.)
    target_img = torch.clamp(target_img, min=0, max=1.)

    if len(input_img) == 3:
        input_img = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()
        target_img = target_img.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        input_img = input_img.squeeze().cpu().numpy()
        pred_img = pred_img.squeeze().cpu().numpy()
        target_img = target_img.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(save_cfg['name'], fontsize=20)
    ax[0].imshow(input_img, cmap='gray')
    ax[0].set_title("Input", fontsize=20)
    ax[0].grid(False)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n".format(save_cfg['org_psnr'], save_cfg['org_ssim']), fontsize=20)

    ax[1].imshow(pred_img, cmap='gray')
    ax[1].set_title("Prediction", fontsize=20)
    ax[1].grid(False)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n".format(save_cfg['pred_psnr'], save_cfg['pred_ssim']), fontsize=20)

    ax[2].imshow(target_img, cmap='gray')
    ax[2].set_title("Target", fontsize=20)
    ax[2].grid(False)
    fig.savefig(os.path.join(save_cfg['save_path'], 'result_{}.png'.format(save_cfg['name'])))
    plt.close()


def plot_save_ct(input_img, pred_img, target_img, save_cfg):
    input_img = torch.clamp(input_img, min=0, max=1.)
    pred_img = torch.clamp(pred_img, min=0, max=1.)
    target_img = torch.clamp(target_img, min=0, max=1.)

    if len(input_img) == 3:
        input_img = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()
        target_img = target_img.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        input_img = input_img.squeeze().cpu().numpy()
        pred_img = pred_img.squeeze().cpu().numpy()
        target_img = target_img.squeeze().cpu().numpy()

    input_img = trunc_denorm(input_img, trunc_max=240, trunc_min=-160)
    pred_img = trunc_denorm(pred_img, trunc_max=240, trunc_min=-160)
    target_img = trunc_denorm(target_img, trunc_max=240, trunc_min=-160)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(save_cfg['name'], fontsize=20)
    ax[0].imshow(input_img, cmap='gray')
    ax[0].set_title("Input", fontsize=20)
    ax[0].grid(False)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n".format(save_cfg['org_psnr'], save_cfg['org_ssim']), fontsize=20)

    ax[1].imshow(pred_img, cmap='gray')
    ax[1].set_title("Prediction", fontsize=20)
    ax[1].grid(False)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n".format(save_cfg['pred_psnr'], save_cfg['pred_ssim']), fontsize=20)

    ax[2].imshow(target_img, cmap='gray')
    ax[2].set_title("Target", fontsize=20)
    ax[2].grid(False)
    fig.savefig(os.path.join(save_cfg['save_path'], 'plot_{}.png'.format(save_cfg['name'])))
    plt.close()


def save_single_slice(pred_img, save_cfg):
    pred_img = torch.clamp(pred_img, min=0, max=1.)

    if len(pred_img) == 3:
        pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        pred_img = pred_img.squeeze().cpu().numpy()

    pred_img = trunc_denorm(pred_img) + 1024
    pred_img = ((pred_img / 4096) * 65535).astype(np.uint16)
    imageio.imwrite(os.path.join(save_cfg['save_path'], 'single_{}.png'.format(save_cfg['name'])), pred_img)


def trunc_denorm(image, trunc_max=3072.0, trunc_min=-1024.0, norm_range_max=3072.0, norm_range_min=-1024.0):
    image = denormalize(image, norm_range_max, norm_range_min)
    image = trunc(image, trunc_max, trunc_min)
    return image


def denormalize(image, norm_range_max=3072.0, norm_range_min=-1024.0):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image


def trunc(image, trunc_max, trunc_min):
    image[image <= trunc_min] = trunc_min
    image[image >= trunc_max] = trunc_max
    return image


def save_nifti(arrays, save_cfg, params):
    nifti_org = sorted(glob.glob(os.path.join(params.data_dir, params.subject, '*.nii.gz')))
    image_org = sitk.ReadImage(nifti_org[0])
    images = sitk.GetImageFromArray(arrays)
    images.CopyInformation(image_org)
    sitk.WriteImage(images, os.path.join(save_cfg['save_path'], 'result_{}_{}_{}epoch.nii.gz'.format(params.desc, params.subject, params.test_epoch)))
