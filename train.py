import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import AAPM
from models import model_selector, loss_selector
from prep import load_dataset
from utils.logging import init_csv, write_csv, save_args
from utils.logging import write_value_tb, write_images_tb, write_hparams_tb
from utils.measures import compute_measure_torch
from utils.plot_img import save_figure


def train(args):
    start = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    checkpoints_path = os.path.join(args.checkpoints_path, args.model, args.fold, start)
    tensorboard_path = os.path.join(args.tensorboard_path, args.model, args.fold, start)
    results_path = os.path.join(args.results_path, args.model, args.fold, start)

    args.checkpoints_path = checkpoints_path
    args.tensorboard_path = tensorboard_path
    args.results_path = results_path

    train_log = os.path.join(results_path, args.model + "_train_" + start + '.csv')

    # Check directories
    os.makedirs(checkpoints_path, exist_ok=True)
    print('Create path: {}'.format(checkpoints_path))

    os.makedirs(tensorboard_path, exist_ok=True)
    print('Create path: {}'.format(tensorboard_path))

    os.makedirs(results_path, exist_ok=True)
    print('Create path: {}'.format(results_path))
    print()

    # Initialize and Save train log (txt, csv, etc)
    save_args(args=args, save_path=results_path, time=start)
    init_csv(csv_file=train_log, init_header=['Epochs', 'PSNR', 'SSIM', 'RMSE'])

    # Load datasets
    trainset = load_dataset(path=args.data_path, fold=args.fold, phase="train")
    validset = load_dataset(path=args.data_path, fold=args.fold, phase="valid")
    print("Train: {} | Test: {}\n".format(len(trainset["full"]), len(validset["full"])))

    # Dataloader
    dataset_train = AAPM(mode="train", dataset=trainset, patch_size=args.patch_size, num_patches=args.num_patches)
    dataset_valid = AAPM(mode="test", dataset=validset, patch_size=args.patch_size, num_patches=args.num_patches)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1, shuffle=False, num_workers=args.num_workers)
    length_train, length_valid = len(dataset_train), len(dataset_valid)
    iter_print = int(length_train / 5)

    # Model
    model = model_selector(args)
    criterion = loss_selector()
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    # Hyper-Parameters
    learning_rate = args.learning_rate
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True, threshold=1e-7, threshold_mode='abs')

    # Tensorboard
    writer = SummaryWriter(log_dir=tensorboard_path, filename_suffix='-' + start)
    writer_valid = SummaryWriter(log_dir=tensorboard_path + '/valid', filename_suffix='-' + start)

    # Training Loop
    n_iters = 0
    best_psnr, best_ssim, best_rmse = 0.0, 0.0, 0.0
    start_time = time.time()
    for epoch in range(args.num_epochs):
        train_loss, iter_loss, epoch_loss = 0.0, 0.0, 0.0
        model.train()
        for iter_train, (_, x_train, y_train) in enumerate(dataloader_train):
            inputs, labels = x_train.to(device).squeeze(0), y_train.to(device).squeeze(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            iter_loss += train_loss
            epoch_loss += train_loss
            n_iters += 1

            write_value_tb(writer, train_loss, steps=n_iters, name="loss_iters", group="train")
            if (n_iters + 1) % iter_print == 0:
                print("Epoch: {}/{} | Iter: {} | Loss: {:.6f} | Time: {:.1f}s".format(epoch+1, args.num_epochs, n_iters+1, train_loss, time.time() - start_time))

        print("Epoch {} | Iter: {} | Loss: {:.6f} | Time: {:.1f}s".format(epoch+1, n_iters+1, epoch_loss / (iter_train+1), time.time()-start_time))
        write_value_tb(writer, epoch_loss / (iter_train+1), steps=epoch, name="loss_epoch", group="train")
        # save_figure(inputs, labels, outputs, path=results_path, epochs=epoch+1, phase="train")

        # Validation Loop
        if (epoch + 1) % args.valid_epochs == 0:
            sum_psnr, sum_ssim, sum_rmse = 0, 0, 0
            avg_psnr, avg_ssim, avg_rmse = 0, 0, 0

            sum_psnr_org, sum_ssim_org, sum_rmse_org = 0, 0, 0
            avg_psnr_org, avg_ssim_org, avg_rmse_org = 0, 0, 0

            model.eval()
            for iter_valid, (_, x_valid, y_valid) in enumerate(dataloader_valid):
                inputs, labels = x_valid.to(device), y_valid.to(device)
                inputs, labels = inputs.view(-1, 1, 512, 512), labels.view(-1, 1, 512, 512)

                with torch.no_grad():
                    outputs = model(inputs)

                metric_org, metric_out = compute_measure_torch(inputs, labels, outputs, device, args)
                sum_psnr += metric_out[0]
                sum_ssim += metric_out[1]
                sum_rmse += metric_out[2]
                sum_psnr_org += metric_org[0]
                sum_ssim_org += metric_org[1]
                sum_rmse_org += metric_org[2]

                if iter_valid == length_valid // 2:
                    input_plot, out_plot, label_plot = inputs, outputs, labels

                avg_psnr, avg_ssim, avg_rmse = sum_psnr / (iter_valid + 1), sum_ssim / (iter_valid + 1), sum_rmse / (iter_valid + 1)
                avg_psnr_org, avg_ssim_org, avg_rmse_org = sum_psnr_org / (iter_valid + 1), sum_ssim_org / (iter_valid + 1), sum_rmse_org / (iter_valid + 1)

            write_value_tb(writer, avg_psnr, steps=epoch, name="PSNR", group="measures")
            write_value_tb(writer, avg_ssim, steps=epoch, name="SSIM", group="measures")
            write_value_tb(writer, avg_rmse, steps=epoch, name="RMSE", group="measures")

            write_value_tb(writer_valid, avg_psnr_org, steps=epoch, name="PSNR", group="measures")
            write_value_tb(writer_valid, avg_ssim_org, steps=epoch, name="SSIM", group="measures")
            write_value_tb(writer_valid, avg_rmse_org, steps=epoch, name="RMSE", group="measures")

            write_images_tb(writer, [input_plot, out_plot, label_plot], steps=epoch, name="input_pred_label", group=start)
            write_csv(csv_file=train_log, log=[epoch, avg_psnr, avg_ssim, avg_rmse], log_type="append")
            save_figure(input_plot, out_plot, label_plot, path=results_path, epochs=epoch+1, index=iter_valid+1, phase="valid")
            scheduler.step(avg_psnr)

            print("Testing on {} | PSNR: {:.4f} | SSIM: {:.4f} | RMSE: {:.4f} | Time: {:.1f}s\n".format(epoch+1, avg_psnr, avg_ssim, avg_rmse, time.time()-start_time))

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch_psnr = epoch + 1
                model_path = os.path.join(checkpoints_path, args.model + "_best_psnr.pth")
                print("Save model to {}\n".format(model_path))
                torch.save(model.state_dict(), model_path)
                if avg_ssim > best_ssim:
                    best_ssim = avg_ssim
                    best_rmse = avg_rmse
                    best_epoch_ssim = epoch + 1
                    model_path = os.path.join(checkpoints_path, args.model + "_best.pth")
                    print("Save model to {}\n".format(model_path))
                    torch.save(model.state_dict(), model_path)

        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        write_value_tb(writer, lr, steps=epoch, name="learning_rate", group="train")

        if (epoch + 1) % args.save_epochs == 0:
            model_path = os.path.join(checkpoints_path, args.model + "_epoch" + str(epoch + 1) + ".pth")
            print("Save model to {}\n".format(model_path))
            torch.save(model.state_dict(), model_path)

    write_hparams_tb(writer, args, metrics=[best_psnr, best_ssim, best_rmse, best_epoch_psnr, best_epoch_ssim])
    writer.close()
