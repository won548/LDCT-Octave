import torch
import torch.nn as nn
from thop import profile

# https://github.com/SSinyu/RED-CNN
class REDCNN(nn.Module):
    def __init__(self, out_ch=64, kernel_size=3):
        super(REDCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=kernel_size, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out


def init_weights(model):
    if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(model.weight, mean=0., std=0.01)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


if __name__ == '__main__':
    x = torch.ones((16, 1, 64, 64))
    model = REDCNN()
    z = model(x)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = model.to(device)
    x = x.to(device)
    total_ops, total_params = profile(model, (x,), verbose=False)

    print()
    print("Params(M): {:.4f} ".format(total_params / (1000 ** 2)))
    print(" FLOPs(G): {:.2f}".format(total_ops / (1000 ** 3)))
    print("Output shape:", z.shape)
    print()