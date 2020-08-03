import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4)
        return out


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), self.vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)
        return out


class PerceptualLoss(torch.nn.Module):
    def __init__(self, network, device):
        super(PerceptualLoss, self).__init__()
        self.device = "cuda:" + device
        if network == "VGG16":
            self.network = VGG16(requires_grad=False).to(self.device)
        elif network == "VGG19":
            self.network = VGG19(requires_grad=False).to(self.device)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = 0.0
        labels = torch.cat((labels, labels, labels), 1)
        inputs = torch.cat((inputs, inputs, inputs), 1)
        features_y = self.network(labels)
        features_x = self.network(inputs)

        for name in features_x._fields:
            out += self.loss(getattr(features_y, name), getattr(features_x, name)) / 3

        return out


if __name__ == "__main__":
    # net = VGG19()
    x = torch.rand((1, 1, 64, 64)).to("cuda:0")
    y = torch.rand((1, 1, 64, 64)).to("cuda:0")
    loss_P_VGG16 = PerceptualLoss(network="VGG16", device="0")
    loss_P = loss_P_VGG16(x, y)
    print(loss_P)
    # net = VGG16()
    # print(net)
