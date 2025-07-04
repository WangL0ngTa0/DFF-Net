import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataloader
import model
import Myloss
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DFF_net = model.enhance_net_nopool().cuda()

    DFF_net.apply(weights_init)
    if config.load_pretrain:
        DFF_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)


    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()
    L_lp = Myloss.LPIPSloss(device='cuda')

    optimizer = torch.optim.Adam(DFF_net.parameters(), lr=config.lr, weight_decay=config.weight_decay, amsgrad=False)
   

    
    DFF_net.train()
    writer = SummaryWriter()
    min_loss = float('inf')
    best_epoch = 0

    for epoch in range(config.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, config.num_epochs))
        loss_list = []
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhance_image_1,enhance_image, r = DFF_net(img_lowlight)

            Loss_TV = 200 * L_TV(r)
            loss_spa = 1 * torch.mean(L_spa(enhance_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhance_image))
            loss_exp = 10 * torch.mean(L_exp(enhance_image))
            loss_lp = 1 * torch.mean(L_lp(enhance_image, img_lowlight))

            loss = Loss_TV + loss_spa +  loss_exp + loss_col + loss_lp
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DFF_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration {}: {}".format(iteration + 1, loss.item()))
            if ((iteration + 1) % config.snapshot_iter) == 0:
                writer.add_scalar('Loss/train', loss, 500 * epoch + iteration + 1)
                torch.save(DFF_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        epoch_loss = np.mean(loss_list)
        writer.add_scalar('Loss/train-epoch-mean', epoch_loss, epoch + 1)
        print("Average loss for epoch {}: {}".format(epoch + 1, epoch_loss))

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_epoch = epoch + 1

    print("Training completed.")
    print("Minimum loss: {} at epoch {}".format(min_loss, best_epoch))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
