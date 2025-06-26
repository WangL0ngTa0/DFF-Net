import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Myattention
from DRBlock import DRB

class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=1, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
        self.LRelu = nn.LeakyReLU(inplace=True)

        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.da1 = Myattention.DoubleAttentionModule(32)
        self.DRB1 = DRB(in_feature = 32, out_feature = 32)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.da2 = Myattention.DoubleAttentionModule(64)
        self.DRB2 = DRB(in_feature = 64, out_feature = 64)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.da3 = Myattention.DoubleAttentionModule(128)
        self.DRB3 = DRB(in_feature = 128, out_feature = 128)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.da4 = Myattention.DoubleAttentionModule(256)
        self.DRB4 = DRB(in_feature = 256, out_feature = 256)

        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv2_1_down = nn.Conv2d(64, 32, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3_1_down = nn.Conv2d(128, 32, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(128, 32, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(128, 32, 1, padding=0)
        self.conv4_1_down = nn.Conv2d(256, 32, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(256, 32, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(256, 32, 1, padding=0)
        self.conv5_1_down = nn.Conv2d(256, 32, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(256, 32, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(256, 32, 1, padding=0)

        self.score_dsn1 = nn.Conv2d(32, 24, 1)
        self.score_dsn2 = nn.Conv2d(32, 24, 1)
        self.score_dsn3 = nn.Conv2d(32, 24, 1)
        self.score_dsn4 = nn.Conv2d(32, 24, 1)
        self.score_dsn5 = nn.Conv2d(32, 24, 1)
        self.cat1 = nn.Conv2d(64, 32, 1)
        self.cat2 = nn.Conv2d(96, 32, 1)
        self.bottle = nn.Conv2d(120, 24, 1)

    def forward(self, x):
        # size
        img_H, img_W = x.shape[2], x.shape[3]
        # base_net
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)
        pool1 = self.LRelu(self.DRB1(pool1))
        pool1 = self.DRB1(pool1)
        pool1_1 = self.da1(pool1)

        conv2_1 = self.relu(self.conv2_1(pool1_1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)
        pool2 = self.LRelu(self.DRB2(pool2))
        pool2 = self.DRB2(pool2)
        pool2_1 = self.da2(pool2)

        conv3_1 = self.relu(self.conv3_1(pool2_1))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))        
        pool3 = self.maxpool(conv3_3)
        pool3 = self.LRelu(self.DRB3(pool3))
        pool3 = self.DRB3(pool3)
        pool3_1 = self.da3(pool3)

        conv4_1 = self.relu(self.conv4_1(pool3_1))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool(conv4_3)
        pool4 = self.LRelu(self.DRB4(pool4))
        pool4 = self.DRB4(pool4)
        pool4_1 = self.da4(pool4)

        conv5_1 = self.relu(self.conv5_1(pool4_1))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        cat1 = torch.cat([conv1_1, conv1_2], 1)
        cat1_out = self.cat1(cat1)
        cat2 = torch.cat([conv2_1_down, conv2_2_down], 1)
        cat2_out = self.cat1(cat2)
        cat3 = torch.cat([conv3_1_down, conv3_2_down, conv3_3_down], 1)
        cat3_out = self.cat2(cat3)
        cat4 = torch.cat([conv4_1_down, conv4_2_down, conv4_3_down], 1)
        cat4_out = self.cat2(cat4)
        cat5 = torch.cat([conv5_1_down, conv5_2_down, conv5_3_down], 1)
        cat5_out = self.cat2(cat5)

        so1_out = self.score_dsn1(cat1_out)
        so2_out = self.score_dsn2(cat2_out)
        so3_out = self.score_dsn3(cat3_out)
        so4_out = self.score_dsn4(cat4_out)
        so5_out = self.score_dsn5(cat5_out)

        weight_deconv2 = make_bilinear_weights(2, 24).cuda()
        weight_deconv3 = make_bilinear_weights(4, 24).cuda()
        weight_deconv4 = make_bilinear_weights(8, 24).cuda()
        weight_deconv5 = make_bilinear_weights(16, 24).cuda()

        upsample2 = F.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = F.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = F.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = F.conv_transpose2d(so5_out, weight_deconv5, stride=16)
      
        # center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        fuse_img_1 = torch.cat([so1, so2, so3, so4, so5], 1)
        fuse_img_2 = torch.tanh(self.bottle(fuse_img_1))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(fuse_img_2, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)		
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)		
        x = x + r6 * (torch.pow(x, 2) - x)	
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r

    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]

def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w
