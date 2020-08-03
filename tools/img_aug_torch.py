import random

import PIL.Image
import numpy as np
import torchvision.transforms.functional as TF


def toPIL(img):
    img_PIL = TF.to_pil_image(img, mode='F')
    return img_PIL


def toTensor(img):
    img = TF.to_tensor(img)
    return img


def random_scale(input_img, target_img, min, max, p):
    if random.random() > p:
        width, height = input_img.size
        target_scale = random.uniform(min, max)
        input_img = TF.resize(input_img, (int(width * target_scale), int(height * target_scale)), interpolation=PIL.Image.NEAREST)
        target_img = TF.resize(target_img, (int(width * target_scale), int(height * target_scale)), interpolation=PIL.Image.NEAREST)

    return input_img, target_img


def random_rotate(input_img, label_img, min, max, p):
    if random.random() > p:
        target_angle = random.uniform(min, max)
        input_img = TF.rotate(input_img, target_angle, resample=PIL.Image.BILINEAR, expand=False)
        label_img = TF.rotate(label_img, target_angle, resample=PIL.Image.BILINEAR, expand=False)

    return input_img, label_img


def random_flip(input_img, label_img, p):
    if random.random() > p:
        target_flip = np.random.choice(['vertical', 'horizontal'], 1)
        if target_flip == 'vertical':
            input_img = TF.vflip(input_img)
            label_img = TF.vflip(label_img)
        elif target_flip == 'horizontal':
            input_img = TF.hflip(input_img)
            label_img = TF.hflip(label_img)

    return input_img, label_img


def random_crop(input_img, label_img, patch_size):
    width, height = input_img.size
    top = np.random.randint(0, height - patch_size)
    left = np.random.randint(0, width - patch_size)

    input_img = TF.crop(input_img, top, left, patch_size, patch_size)
    label_img = TF.crop(label_img, top, left, patch_size, patch_size)

    return input_img, label_img


def random_patches(input_img, label_img, patch_size, num_patches):
    input_list, label_list = [], []
    for n in range(num_patches):
        input_patch, label_patch = random_crop(input_img, label_img, patch_size)
        input_patch, label_patch = np.array(input_patch), np.array(label_patch)
        input_patch, label_patch = np.expand_dims(input_patch, axis=0), np.expand_dims(label_patch, axis=0)
        input_list.append(input_patch)
        label_list.append(label_patch)
    input_patches, label_patches = np.concatenate(input_list, axis=0), np.concatenate(label_list, axis=0)
    input_patches, label_patches = np.expand_dims(input_patches, axis=1), np.expand_dims(label_patches, axis=1)

    return input_patches, label_patches
