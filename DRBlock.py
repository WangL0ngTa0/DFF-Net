import torch
import torch.nn as nn
import numpy as np

class laplacian(nn.Module):
    def __init__(self, channels):
        super(laplacian, self).__init__()
        laplacian_filter = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        self.conv_x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(torch.from_numpy(laplacian_filter))
        self.conv_y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_y.weight.data.copy_(torch.from_numpy(laplacian_filter.T))
    def forward(self, x):
        laplacianx = self.conv_x(x)
        laplaciany = self.conv_y(x)
        x = torch.abs(laplacianx) + torch.abs(laplaciany)
        return x

class Sobelxy(nn.Module):
    def __init__(self, channels):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.conv_x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.conv_y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_y.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.conv_x(x)
        sobely = self.conv_y(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x
    
class DRB(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(DRB, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_feature, out_channels=2 * in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2_1 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv2_2 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=5, stride=1, padding=2, bias=True)
        self.Conv2_3 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=7, stride=1, padding=3, bias=True)
        self.laplacian = laplacian(2 * in_feature)
        self.sobel = Sobelxy(2 * in_feature)
        self.Conv3 = nn.Conv2d(in_channels=2 * in_feature, out_channels=in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv_end = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.LRelu = nn.LeakyReLU()
    def forward(self, x):
        out1 = self.LRelu(self.Conv1(x))

        out21 = self.LRelu(self.Conv2_1(out1))
        out22 = self.LRelu(self.Conv2_2(out1))
        out23 = self.LRelu(self.Conv2_3(out1))
        out24 = self.sobel(out1)
        out25 = self.sobel(out1)
        out3 = torch.add(out1, (out21 + out22 + out23 + out24 + out25))
        out4 = self.LRelu(self.Conv3(out3))
        out5 = torch.add(x, out4)
        out6 = self.LRelu(self.Conv_end(out5))
        return out6