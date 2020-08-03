import csv
import os

import torch
import torchvision

import utils.conversion as convert


def save_args(args, save_path, time):
    def print_log(*argv, path):
        if isinstance(argv, str):
            string = argv
        else:
            string = ' '.join([str(arg) for arg in argv])

        # write to file
        with open(os.path.join(path, time + ".txt"), 'at') as wf:
            wf.write(string + '\n')

        # print stdio
        print(string)

    print_log('=' * 40, path=save_path)
    print_log(' ' * 14 + 'Arguments', path=save_path)
    for arg in sorted(vars(args)):
        print_log(arg + ':', getattr(args, arg), path=save_path)
    print_log('=' * 40, path=save_path)


def init_csv(csv_file, init_header, log_type="append"):
    if log_type == "append":
        mode = "a"
    elif log_type == "write":
        mode = "w"
    else:
        mode = "w"

    with open(csv_file, mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(init_header)


def write_csv(csv_file, log, log_type="append"):
    if log_type == "append":
        mode = "a"
    elif log_type == "write":
        mode = "w"
    else:
        mode = "w"

    with open(csv_file, mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log)


def write_value_tb(writer, value, steps, name, group=None):
    writer.add_scalar(group + "/" + name, value, steps + 1)


def write_images_tb(writer, images, steps, name, group=None, center=True):
    img_input = convert.trunc_denorm(images[0], trunc_min=-160, trunc_max=240)
    img_pred = convert.trunc_denorm(images[1], trunc_min=-160, trunc_max=240)
    img_label = convert.trunc_denorm(images[2], trunc_min=-160, trunc_max=240)

    input_grid = torchvision.utils.make_grid(img_input.detach(), nrow=1, normalize=True)
    pred_grid = torchvision.utils.make_grid(img_pred.detach(), nrow=1, normalize=True)
    label_grid = torchvision.utils.make_grid(img_label.detach(), nrow=1, normalize=True)
    visual_grid = torch.cat((input_grid, pred_grid, label_grid), dim=2)

    writer.add_image(group + "/" + name, visual_grid, steps + 1)


def write_hparams_tb(writer, args, metrics):
    hparams = {"model": args.model, "batch_size": args.batch_size, "num_patches": args.num_patches, "lr": args.learning_rate,
               "channels": args.channels, "kernel_size": args.kernel_size, "alpha": args.alpha, "patch_size": args.patch_size}
    metrics = {"best/pnsr": metrics[0], "best/ssim": metrics[1], "best/rmse": metrics[2], "best/epoch_psnr": metrics[3], "best/epoch_ssim": metrics[4]}

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)


