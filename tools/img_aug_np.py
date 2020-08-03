import numpy as np
import skimage.transform


def random_rotate(input_img, label_img, angle, p):
    if np.random.uniform(0, 1) > p:
        target_angle = np.random.choice(angle, 1)
        input_img = skimage.transform.rotate(input_img, target_angle, resize=False, center=(input_img.shape[0]/2, input_img.shape[1]/2))
        label_img = skimage.transform.rotate(label_img, target_angle, resize=False, center=(input_img.shape[0]/2, input_img.shape[1]/2))

    return input_img, label_img


def random_flip(input_img, label_img, p):
    if np.random.uniform(0, 1) > p:
        target_flip = np.random.choice(['vertical', 'horizontal'], 1)
        if target_flip == 'vertical':
            input_img = np.flipud(input_img).copy()
            label_img = np.flipud(label_img).copy()
        elif target_flip == 'horizontal':
            input_img = np.fliplr(input_img).copy()
            label_img = np.fliplr(label_img).copy()

    return input_img, label_img


def random_scale(input_img, label_img, scale, p):
    if np.random.uniform(0, 1) > p:
        target_scale = np.random.choice(scale, 1)
        target_x, target_y = int(input_img.shape[1] * target_scale), int(input_img.shape[0] * target_scale)
        input_img = skimage.transform.resize(input_img, output_shape=(target_x, target_y))
        label_img = skimage.transform.resize(label_img, output_shape=(target_x, target_y))
    return input_img, label_img


def random_crop(input_img, target_img, patch_size):
    height, width = input_img.shape[0], input_img.shape[1]
    y, x = np.random.randint(0, height - patch_size), np.random.randint(0, width - patch_size)

    input_img = input_img[np.newaxis, y:y+patch_size, x:x+patch_size]
    target_img = target_img[np.newaxis, y:y+patch_size, x:x+patch_size]

    return input_img, target_img


def FOV_random_crop(input_img, target_img, patch_size):
    height, width = input_img.shape[0], input_img.shape[1]
    r = np.sqrt(np.random.uniform(0, 1))
    theta = 2 * np.pi * np.random.random()

    px = r * np.cos(theta)
    py = r * np.sin(theta)

    if py < 0:
        py += 1
    if px < 0:
        px += 1

    x = int(py * (width - patch_size))
    y = int(py * (width - patch_size))
    input_img = input_img[np.newaxis, y:y+patch_size, x:x+patch_size]
    target_img = target_img[np.newaxis, y:y+patch_size, x:x+patch_size]

    return input_img, target_img


def random_patches(input_img, label_img, patch_size, num_patches):
    input_patches, label_patches = [], []
    for n in range(num_patches):
        # input_patch, label_patch = random_crop(input_img, label_img, patch_size)
        input_patch, label_patch = FOV_random_crop(input_img, label_img, patch_size)
        input_patches.append(input_patch)
        label_patches.append(label_patch)
    input_patches, label_patches = np.concatenate(input_patches, axis=0), np.concatenate(label_patches, axis=0)
    return input_patches, label_patches