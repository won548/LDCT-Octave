import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader

from prep import load_dataset
from tools.img_aug_torch import toPIL, random_scale, random_flip, random_rotate, random_patches


class AAPM(Dataset):
    def __init__(self, dataset, mode, patch_size=64, num_patches=64):
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.quarter = dataset['quarter']
        self.full = dataset['full']

    def __len__(self):
        return len(self.quarter)

    def __getitem__(self, idx):
        quarter = pydicom.dcmread(self.quarter[idx])
        full = pydicom.dcmread(self.full[idx])

        subject = self.quarter[idx].split('/')[-1].split('_')[0]
        slice = self.quarter[idx].split('/')[-1].split('.')[3]
        name = subject + '_' + slice

        input_img, label_img = quarter.pixel_array.astype(np.float32), full.pixel_array.astype(np.float32)
        input_img, label_img = np.expand_dims(input_img, axis=2), np.expand_dims(label_img, axis=2)

        if self.mode == "train":
            input_aug, label_aug = toPIL(input_img), toPIL(label_img)
            input_aug, label_aug = random_scale(input_aug, label_aug, min=0.5, max=2.0, p=0.5)
            input_aug, label_aug = random_rotate(input_aug, label_aug, min=-90, max=90, p=0.5)
            input_aug, label_aug = random_flip(input_aug, label_aug, p=0.5)
            input_img, label_img = random_patches(input_aug, label_aug, self.patch_size, self.num_patches)
        elif self.mode == "valid" or self.mode == "test":
            input_img, label_img = input_img.transpose((2, 0, 1)), label_img.transpose((2, 0, 1))

        input_img = torch.from_numpy(input_img)
        label_img = torch.from_numpy(label_img)

        input_img = input_img / 4095.
        label_img = label_img / 4095.

        return dict(source=input_img, target=label_img, name=name)


def main():
    batch_size = 4
    num_patches = 4
    if num_patches > 1:
        batch_size = 1

    trainset = load_dataset(path="/home/dongkyu/Datasets/AAPM", fold=10, phase='train')
    testset = load_dataset(path="/home/dongkyu/Datasets/AAPM", fold=10, phase='test')

    dataset_train = AAPM(mode='train', dataset=trainset, patch_size=64, num_patches=num_patches)
    dataset_valid = AAPM(mode='valid', dataset=testset, patch_size=64, num_patches=num_patches)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1, shuffle=False, num_workers=4)

    while True:
        for iter, batches in enumerate(dataloader_train):
            x, y = batches['source'].to('cuda:0'), batches['target'].to('cuda:0')
            if num_patches > 1:
                x, y = x.squeeze(0), y.squeeze(0)
            else:
                x, y = x.squeeze(1), y.squeeze(1)
            print(x.shape)
            plt.figure()
            for i in range(x.shape[0]):
                plt.subplot(2, 2, i+1)
                plt.imshow(x[i].squeeze().cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.show()


if __name__ == "__main__":
    print('AAPM_loader.py')
    main()
