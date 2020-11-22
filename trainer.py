import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from models import model_selector, loss_selector
from utils.evaluation import compute_measure_torch
from utils.logging import init_csv, write_csv
from utils.save_img import plot_save_ct, trunc_denorm, save_nifti


class Trainer(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""
    def __init__(self, params, mode):
        # Model
        self.model_name = params.model
        self.model = model_selector(params)
        self.params = params

        if mode == 'train':
            # Training hyper-parameters
            self.step = 0
            self.num_epochs = params.num_epochs
            self.val_epoch = params.val_epoch
            self.batch_size = params.batch_size
            self.num_patches = params.num_patches
            self.learning_rate = params.learning_rate
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=params.adam[:2], eps=params.adam[2])
            self.scheduler = params.scheduler
            if self.scheduler:
                self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.num_epochs // 10, factor=0.5, verbose=True)
            self.criterion = loss_selector(params)

            # Create directories
            self.train_date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            self.test_fold = params.test_fold
            self.exp_dir = os.path.join(params.base_dir, self.model_name, self.test_fold, self.train_date)
            self.checkpoints_path = os.path.join(self.exp_dir, 'checkpoints')
            self.results_path = os.path.join(self.exp_dir, 'results')
            self.create_directories()

            self.train_log = os.path.join(self.exp_dir, "train_log_" + self.train_date + '.csv')
            self.param_txt = os.path.join(self.exp_dir, "train_params_" + self.train_date + '.txt')
            self.print_params()
            init_csv(csv_file=self.train_log, init_header=['Epochs', 'PSNR', 'SSIM', 'RMSE'])

            # Evaluation Metric
            self.best_psnr = 0.0

            # Device settings
            self.device = torch.device("cuda:" + params.device if torch.cuda.is_available() else "cpu")
            if 'cuda' in str(self.device):
                self.model.to(self.device)
                self.criterion.to(self.device)
        else:
            self.desc = params.desc
            self.test_date = params.test_date
            self.test_fold = params.test_fold
            self.test_epoch = params.test_epoch
            self.exp_dir = os.path.join(params.base_dir, self.model_name, self.test_fold, self.test_date)
            self.checkpoints_path = os.path.join(self.exp_dir, 'checkpoints')
            self.results_path = os.path.join(self.exp_dir, 'results')
            self.save_path = os.path.join(self.results_path, self.desc)

            os.makedirs(self.save_path, exist_ok=True)
            print('Create path: {}'.format(self.save_path))
            print()

            self.device = torch.device("cuda:" + params.device if torch.cuda.is_available() else "cpu")
            if self.test_epoch == 'best':
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, "model_best.pth"), map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_path, "model_epoch" + self.test_epoch + ".pth"), map_location=self.device))
            if 'cuda' in str(self.device):
                self.model.to(self.device)
            self.model.eval()

        if self.params.model == 'redcnn':
            print('REDCNN (Chen et al., IEEE TMI 2018)')
        elif self.params.model == 'cnn_oct':
            print('LDCT using Octave convolution (Won et al., MICCAI PRIME 2020)')
        elif self.params.model == 'unet':
            print('U-Net (Ronneberger et al., arXiv 2015)')

    def train(self, dataloader_train, dataloader_valid):
        start_time = time.time()
        for epoch in range(self.num_epochs):
            for idx, batches in enumerate(dataloader_train):
                inputs, labels = batches['source'].to(self.device), batches['target'].to(self.device)
                if self.num_patches > 1:
                    inputs, labels = inputs.squeeze(0), labels.squeeze(0)
                else:
                    inputs, labels = inputs.squeeze(1), labels.squeeze(1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.step += 1
                if (idx + 1) % (len(dataloader_train) // 5) == 0:
                    print("Epoch: {}/{} | Step: {} | Loss: {:.9f} | Time: {:.1f}s".format(epoch + 1, self.num_epochs, self.step, loss, time.time() - start_time))

            model_path = os.path.join(self.checkpoints_path, "model_epoch{}.pth".format(epoch+1))
            torch.save(self.model.state_dict(), model_path)

            if (epoch + 1) % self.val_epoch == 0:
                self.model.eval()
                self.valid(dataloader_valid, epoch)
                self.model.train()

    def valid(self, dataloader_valid, epoch):
        print("Validation...")
        loss_valid = []
        ori_psnr, ori_ssim, org_rmse = [], [], []
        pred_psnr, pred_ssim, pred_rmse = [], [], []
        self.model.eval()
        with torch.no_grad():
            for idx, batches in tqdm(enumerate(dataloader_valid)):
                inputs, labels = batches['source'].to(self.device), batches['target'].to(self.device)
                outputs = self.model(inputs)

                loss_valid.append(self.criterion(outputs, labels).item())
                original_result, pred_result = compute_measure_torch(inputs, labels, outputs)
                ori_psnr.append(original_result[0])
                ori_ssim.append(original_result[1])
                org_rmse.append(original_result[2])
                pred_psnr.append(pred_result[0])
                pred_ssim.append(pred_result[1])
                pred_rmse.append(original_result[2])

                if len(dataloader_valid) // 2 == idx:
                    inputs_plot = inputs
                    outputs_plot = outputs.detach()
                    labels_plot = labels

        loss_avg = sum(loss_valid) / len(loss_valid)
        pred_psnr_avg = sum(pred_psnr) / len(pred_psnr)
        pred_ssim_avg = sum(pred_ssim) / len(pred_ssim)
        print("Epoch {} | Loss: {:.9f} | PSNR: {:.4f} | SSIM: {:.4f}\n".format(epoch + 1, loss_avg, pred_psnr_avg, pred_ssim_avg))

        if pred_psnr_avg > self.best_psnr:
            self.best_psnr = pred_psnr_avg
            model_path = os.path.join(self.checkpoints_path, "model_best.pth")
            torch.save(self.model.state_dict(), model_path)
            print("Save model to {}\n".format("model_best.pth"))
            write_csv(csv_file=self.train_log, log=[epoch+1, pred_psnr_avg, pred_ssim_avg, '^'], log_type="append")
        else:
            write_csv(csv_file=self.train_log, log=[epoch+1, pred_psnr_avg, pred_ssim_avg], log_type="append")

        if self.scheduler:
            self.scheduler.step(self.best_psnr)

        save_cfg = {'name': str(epoch), 'save_path': self.results_path,
                    'org_psnr': original_result[0], 'org_ssim': original_result[1],
                    'pred_psnr': pred_result[0], 'pred_ssim': pred_result[1]}
        plot_save_ct(inputs_plot, outputs_plot, labels_plot, save_cfg)

    def test(self, dataloader_test):
        pred_nifti = np.zeros(shape=(len(dataloader_test), 512, 512), dtype=np.int16)
        ori_psnr, ori_ssim, org_rmse = [], [], []
        pred_psnr, pred_ssim, pred_rmse = [], [], []
        self.model.eval()
        with torch.no_grad():
            for idx, batches in enumerate(dataloader_test):
                inputs, labels = batches['source'].to(self.device), batches['target'].to(self.device)
                name = batches['name'][0]
                inputs, pad = self.pad(inputs)
                outputs = self.model(inputs)
                inputs, outputs = self.remove_pad(inputs, pad), self.remove_pad(outputs, pad)

                original_result, pred_result = compute_measure_torch(inputs, labels, outputs)
                ori_psnr.append(original_result[0])
                ori_ssim.append(original_result[1])
                org_rmse.append(original_result[2])
                pred_psnr.append(pred_result[0])
                pred_ssim.append(pred_result[1])
                pred_rmse.append(original_result[2])

                save_cfg = {'name': '{}_{}'.format(str(self.desc), str(name)), 'save_path': self.save_path,
                            'org_psnr': original_result[0], 'org_ssim': original_result[1],
                            'pred_psnr': pred_result[0], 'pred_ssim': pred_result[1]}

                pred_nifti[len(dataloader_test) - idx - 1] = trunc_denorm(outputs.squeeze().cpu())
                # save_single_slice(outputs, save_cfg)
                plot_save_ct(inputs, outputs, labels, save_cfg)
                print("{} | PSNR: {:.4f} | SSIM: {:.4f}".format(idx + 1, pred_result[0], pred_result[1]))

        save_nifti(pred_nifti, save_cfg, self.params)
        pred_psnr_avg = sum(pred_psnr) / len(pred_psnr)
        pred_ssim_avg = sum(pred_ssim) / len(pred_ssim)

        print("Avg PSNR: {:.4f} | Avg SSIM: {:.4f}\n".format(pred_psnr_avg, pred_ssim_avg))

    def create_directories(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        print('Create path: {}'.format(self.exp_dir))

        os.makedirs(self.checkpoints_path, exist_ok=True)
        print('Create path: {}'.format(self.checkpoints_path))

        os.makedirs(self.results_path, exist_ok=True)
        print('Create path: {}'.format(self.results_path))
        print()

    def print_params(self):
        """Formats parameters to print when training."""
        param_dict = vars(self.params)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('Training parameters: ')
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()

        param_list = []
        param_list.append('Training parameters: \n')
        param_list.append('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        with open(self.param_txt, 'at') as w:
            for param in param_list:
                w.write(param)

    @staticmethod
    def pad(input):
        pad = [0, 0, 0, 0]
        _, c, h, w = input.shape
        pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
        if pw != 0 or ph != 0:
            if pw % 2 == 0 and ph % 2 == 0:
                pad = [pw//2, pw//2, ph//2, ph//2]
            else:
                if ph % 2 != 0 and pw % 2 == 0:
                    pad = [pw//2, pw//2, ph//2, ph//2 + 1]
                elif pw % 2 != 0 and ph % 2 == 0:
                    pad = [pw//2, pw//2 + 1, ph//2, ph//2]
                else:
                    pad = [pw//2, pw//2 + 1, ph//2, ph//2 + 1]
        padded_img = F.pad(input, pad, mode='constant', value=0)
        return padded_img, [pad[2], pad[3], pad[0], pad[1]]

    @staticmethod
    def remove_pad(input, pad):
        _, c, h, w = input.shape
        img = input[:, :, pad[0]:h-pad[1], pad[2]:w-pad[3]]
        return img
