from dataloader import AAPM
from prep import load_dataset
from torch.utils.data import DataLoader


def load_CTDataset(data_dir, params, mode='train'):
    dataset_ct = load_dataset(path=data_dir, fold=params.test_fold, phase=mode)

    if mode == 'train':
        if params.num_patches > 1:
            params.batch_size = 1
        dataset = AAPM(dataset=dataset_ct, mode=mode, patch_size=params.patch_size, num_patches=params.num_patches)
        dataloader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, drop_last=True)
    else:
        dataset = AAPM(dataset=dataset_ct, mode=mode)
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=params.num_workers)

    return dataloader
