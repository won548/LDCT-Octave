import os

import matplotlib.pyplot as plt


def normalize(img, norm_min=0, norm_max=255):
    if list(img.size()) == 4:
        for i in range(img.shape[0]):
            img[i] = (norm_max - norm_min) * img[i]
        img = img.byte()
    elif list(img.size()) == 3:
        img_norm = (norm_max - norm_min) * img
        img = img_norm.byte()
    return img


def save_figure(x, y, y_, path, epochs, index=0, phase="train", norm_min=0, norm_max=255):
    save_path = os.path.join(path, phase)
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(save_path, "Epoch " + str(epochs))
    os.makedirs(save_path, exist_ok=True)

    if x.shape[0] == 1:
        batch_center = 0
    else:
        batch_center = x.shape[0] // 2

    input = x[batch_center]
    # input = normalize(input, norm_min, norm_max)
    input = input.detach().squeeze().cpu().numpy()

    label = y[batch_center]
    # label = normalize(label, norm_min, norm_max)
    label = label.detach().squeeze().cpu().numpy()

    pred = y_[batch_center]
    # pred = normalize(pred, norm_min, norm_max)
    pred = pred.detach().squeeze().cpu().numpy()

    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(input, cmap="gray", vmin=-160, vmax=240)
    ax[0].set_title("Input", fontsize=30)

    ax[1].imshow(pred, cmap="gray", vmin=-160, vmax=240)
    ax[1].set_title('Prediction', fontsize=30)

    ax[2].imshow(label, cmap="gray", vmin=-160, vmax=240)
    ax[2].set_title('Label', fontsize=30)

    f.savefig(os.path.join(save_path, 'result_{}_{}.png'.format(phase, index)))
    plt.close()
