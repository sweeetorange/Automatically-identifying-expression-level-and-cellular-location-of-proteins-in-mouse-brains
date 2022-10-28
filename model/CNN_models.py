# model
from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import random
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import evaluation_test
import gc
from fold5_loaddata import get_loader
from efficientnet_pytorch import EfficientNet

# from sklearn.metrics import f1_score, classification_report, confusion_matrix

writer = SummaryWriter("/log")
cuda_gpu = torch.cuda.is_available()
epochs = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        loss = - ((1 - self.alpha) * ((1 - pt + 1e-5) ** self.gamma) * (target * torch.log(pt + 1e-5)) + self.alpha * (
                (pt + +1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt + 1e-5)))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class BCEFocalLoss_sim(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss_sim, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        loss = - (((1 - pt + 1e-5) ** self.gamma) * (target * torch.log(pt + 1e-5)) + (
                (pt + +1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt + 1e-5)))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# efficient-b0
efficient = EfficientNet.from_pretrained("efficientnet-b0", num_classes=8)

# VGG16
vgg_based = torchvision.models.vgg16(pretrained=True)
'''for param in dense_based.parameters():
    # print(param)
    param.requires_grad = False'''
# Modify the last layer
vgg_based.classifier = torch.nn.Sequential(  # 修改全连接层 自动梯度会恢复为默认值
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 8))

# ResNet18
resnet_based = torchvision.models.resnet18(pretrained=True)
# Modify the last layer
num_ftrs = resnet_based.fc.in_features
resnet_based.fc = nn.Linear(num_ftrs, 8)

# ResNet101
resnet_based = torchvision.models.resnet101(pretrained=True)
# Modify the last layer
num_ftrs = resnet_based.fc.in_features
resnet_based.fc = nn.Linear(num_ftrs, 8)

# DenseNet121
dense_based = torchvision.models.densenet121(pretrained=True)
'''for param in resnet_based.parameters():
    # print(param)
    param.requires_grad = False'''
# Modify the last layer
dense_based.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 8))


def train(model, dataloaders, optimizer, epoch, criterion, scheduler, k):
    sigmoid_fun = nn.Sigmoid()
    since = time.time()
    model.train()
    correct = 0
    for phase in ['train', 'val']:
        loss_own = 0.
        running_precision = 0.
        running_recall = 0.
        running_score = 0.
        batch_num = 0
        if phase == 'train':
            # 学习率更新方式
            scheduler.step()
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
            print('Current learning rate: ' + '%.5f' % cur_lr)
            #  调用模型训练
            model.train()
            # 依次获取所有图像，参与模型训练或测试
            pred_train = []
            ll_train = []
            for batch_idx, (data, label) in enumerate(dataloaders[phase]):
                if (cuda_gpu):
                    data, label = data.cuda(), label.cuda()
                data, label = Variable(data), Variable(label)

                optimizer.zero_grad()    # 把梯度置零
                # calculate loss and metrics
                outputs = model(data)
                # 更改的适合多标签分类的loss和accracy
                loss = criterion(sigmoid_fun(outputs), label)
                # 预测结果的准确性
                precision, recall, f1_score, pred = evaluation_test.calculate_acuracy_mode_one(sigmoid_fun(outputs), label)
                running_precision += precision
                running_recall += recall
                running_score += f1_score
                batch_num += 1
                # backward pass
                loss.backward()    # 反省传播计算得到每个参数的梯度值
                # step
                optimizer.step()   # 梯度下降执行一步参数更新
                loss_own += loss.item()


            epoch_loss = loss_own / len(dataloaders)
            print('{} Loss: {:.4f} '.format("train", epoch_loss))
            epoch_precision = running_precision / batch_num
            print('{} Precision: {:.4f} '.format("train", epoch_precision))
            epoch_recall = running_recall / batch_num
            print('{} Recall: {:.4f} '.format("train", epoch_recall))
            epoch_score = running_score / batch_num
            print('{} f1_score: {:.4f} '.format("train", epoch_score))

            writer.add_scalar('train/train_loss', epoch_loss, epoch)
            writer.add_scalar('train/train_presicion', epoch_precision, epoch)
            writer.add_scalar('train/train_recall', epoch_recall, epoch)
            writer.add_scalar('train/train_score', epoch_score, epoch)

        else:
            with torch.no_grad():
                # 依次获取所有图像，参与模型训练或测试
                model.eval()
                pred_val = []
                ll_val = []
                for batch_idx, (data, label_val) in enumerate(dataloaders[phase]):
                    # 获取输入
                    if (cuda_gpu):
                        data, label_val = data.cuda(), label_val.cuda()
                    data, label_val = Variable(data), Variable(label_val)
                    # 网络前向运行
                    outputs = model(data)
                    # 计算Loss值
                    # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                    loss = criterion(sigmoid_fun(outputs), label_val)
                    precision, recall, f1_score, pred= evaluation_test.calculate_acuracy_mode_one(sigmoid_fun(outputs), label_val)
                    # 计算一个epoch的loss值和准确率
                    loss_own += loss.item()
                    running_precision += precision
                    running_recall += recall
                    running_score += f1_score
                    batch_num += 1

                # calculate loss and error for epoch
                epoch_loss = loss_own / len(dataloaders)
                print('{} Loss: {:.4f} '.format("val", epoch_loss))
                epoch_precision = running_precision / batch_num
                print('{} Precision: {:.4f} '.format("val", epoch_precision))
                epoch_recall = running_recall / batch_num
                print('{} Recall: {:.4f} '.format("val", epoch_recall))
                epoch_score1 = running_score / batch_num
                print('{} f1_score: {:.4f} '.format("val", epoch_score1))

                '''writer.add_scalar('data/train_acc', train_acc, epoch)    # 将我们所需要的数据保存在文件里面供可视化使用(tensorboard)'''
                writer.add_scalar('val/val_loss', epoch_loss, epoch)
                writer.add_scalar('val/val_presicion', epoch_precision, epoch)
                writer.add_scalar('val/val_recall', epoch_recall, epoch)
                writer.add_scalar("val/val_score", epoch_score1,epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_score1


def main():
    start = time.time()
    print("into the main")
    root_dir = "/model"

    n_fold = 1
    for k in range(n_fold):
        dataloaders, dataloaders_test = get_loader(k)
        model = efficient
        if (cuda_gpu):
            model.cuda()
            print("model to gpu", k, "fold")
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999),
                               weight_decay=10e-5)

        scheduler = lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)
        criterion = BCEFocalLoss()

        best_score = 0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            val_acc = train(model, dataloaders, optimizer, epoch, criterion, scheduler, k)
            if val_acc > best_score:
                best_score = val_acc
                best_model = model
                torch.save(best_model, root_dir + "/" + str(k) + "_" + "efficietn_best.pt")
                print("the best model is in ", epoch)

        # torch.save(model.state_dict(), root_dir + "/" + "resnet101_state_tunedivide_8.pt")
        torch.save(model, root_dir + "/" + str(k+5) + "_" + "efficient_fcloss.pt")
        end = time.time()
        print('time taken is ', (end - start))


if __name__ == '__main__':
    main()