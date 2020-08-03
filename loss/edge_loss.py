import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        self.device = "cuda:" + device
        self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float, device=self.device)
        self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float, device=self.device)
        self.sobel_x = self.sobel_x.expand(1, 1, 3, 3)
        self.sobel_y = self.sobel_y.expand(1, 1, 3, 3)

    def forward(self, x, y):
        x_grad_x = F.conv2d(x, self.sobel_x, stride=1, padding=1)
        x_grad_y = F.conv2d(x, self.sobel_y, stride=1, padding=1)
        x_grad = torch.abs(x_grad_x + x_grad_y)

        y_grad_x = F.conv2d(y, self.sobel_x, stride=1, padding=1)
        y_grad_y = F.conv2d(y, self.sobel_y, stride=1, padding=1)
        y_grad = torch.abs(y_grad_x + y_grad_y)

        loss = nn.MSELoss()(y_grad, x_grad)
        return loss
