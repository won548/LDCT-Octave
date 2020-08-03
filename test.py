import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import AAPM
from models import model_selector
from prep import load_dataset
from utils.conversion import trunc_denorm
from utils.logging import init_csv, write_csv
from utils.measures import compute_measure_torch
from utils.plot_img import save_figure
from utils.save_nifti import save_nifti


def load_checkpoint(model, model_path):
    if not os.path.isfile(model_path):
        raise ValueError('Invalid checkpoint file: {}'.format(model_path))

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    # create state_dict
    state_dict = {}

    # convert data_parallal to model
    tmp_state_dict = checkpoint
    for k in tmp_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = tmp_state_dict[k]
        else:
            state_dict[k] = tmp_state_dict[k]

    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Load parameter partially {}, required shape {}, loaded shape {}'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                tmp = torch.zeros(model_state_dict[k].shape)  # create tensor with zero filled
                tmp[:state_dict[k].shape[0]] = state_dict[k]  # fill valid
                state_dict[k] = tmp
        else:
            print('Drop parameter {}'.format(k))

    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}'.format(k))
            state_dict[k] = model_state_dict[k]

    # load state_dict
    model.load_state_dict(state_dict, strict=False)

    return model


def test(args):
    if "best" in args.test_epochs:
        checkpoint_path = os.path.join(args.checkpoints_path, args.model, args.fold, args.test_dates, args.model + "_" + str(args.test_epochs) + '.pth')
    else:
        checkpoint_path = os.path.join(args.checkpoints_path, args.model, args.fold, args.test_dates, args.model + "_epoch" + str(args.test_epochs) + '.pth')
    results_path = os.path.join(args.results_path, args.model, args.fold, args.test_dates)
    test_log = os.path.join(results_path, args.model + "_test_" + args.test_dates + '.csv')

    # Dataloader
    testset = load_dataset(path=args.data_path, fold=args.fold, phase="test")
    print("Test: {}\n".format(len(testset["full"])))

    # Dataloader
    dataset_test = AAPM(mode="test", dataset=testset)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=1)
    length_test = len(dataset_test)

    # Initialize and Save train log (txt, csv, etc)
    init_csv(csv_file=test_log, init_header=['Slice', 'PSNR', 'SSIM', 'RMSE'])

    # Model
    model = model_selector(args)
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    # Metric
    sum_psnr, sum_ssim, sum_rmse = 0, 0, 0
    avg_psnr, avg_ssim, avg_rmse = 0, 0, 0

    sum_psnr_org, sum_ssim_org, sum_rmse_org = 0, 0, 0
    avg_psnr_org, avg_ssim_org, avg_rmse_org = 0, 0, 0

    pred_array = np.zeros((length_test, 512, 512), dtype=np.int16)

    print("Testing...")
    start_time = time.time()
    for iter_test, (name, x, y) in enumerate(dataloader_test):
        img_org = name[0]
        img_org = img_org.split("_")[0]
        inputs, labels = x.to(device), y.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        metric_org, metric_out = compute_measure_torch(inputs, labels, outputs, device, args)
        sum_psnr += metric_out[0]
        sum_ssim += metric_out[1]
        sum_rmse += metric_out[2]
        sum_psnr_org += metric_org[0]
        sum_ssim_org += metric_org[1]
        sum_rmse_org += metric_org[2]

        avg_psnr, avg_ssim, avg_rmse = sum_psnr / (iter_test + 1), sum_ssim / (iter_test + 1), sum_rmse / (iter_test + 1)
        avg_psnr_org, avg_ssim_org, avg_rmse_org = sum_psnr_org / (iter_test + 1), sum_ssim_org / (iter_test + 1), sum_rmse_org / (iter_test + 1)
        print("Iter {}/{} | PSNR: {:.4f} | SSIM: {:.4f} | RMSE: {:.4f}".format(iter_test+1, length_test, avg_psnr, avg_ssim, avg_rmse))

        inputs_norm = trunc_denorm(inputs)
        outputs_norm = trunc_denorm(outputs)
        labels_norm = trunc_denorm(labels)

        save_figure(inputs_norm, labels_norm, outputs_norm, path=results_path, epochs=args.test_epochs, index=iter_test+1, phase="test")
        write_csv(csv_file=test_log, log=[iter_test+1, metric_out[0], metric_out[1], metric_out[2]], log_type="append")

        output_norm = trunc_denorm(outputs.cpu().numpy())
        pred_array[length_test - iter_test - 1] = output_norm

    write_csv(csv_file=test_log, log=["Avg", avg_psnr, avg_ssim, avg_rmse], log_type="append")
    if "best" in args.test_epochs:
        save_nifti(pred_array, img_org, os.path.join(results_path, img_org + "_" + args.model + "_" + str(args.test_epochs) + "_pred.nii.gz"))
    else:
        save_nifti(pred_array, img_org, os.path.join(results_path, img_org + "_" + args.model + "_epoch" + str(args.test_epochs) + "_pred.nii.gz"))
    print("Testing on {} | PSNR: {:.4f} | SSIM: {:.4f} | RMSE: {:.4f} | Time: {:.1f}s\n".format(args.test_epochs, avg_psnr, avg_ssim, avg_rmse, time.time() - start_time))
