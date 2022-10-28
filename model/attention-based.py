from __future__ import print_function

import numpy as np
import imageio
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from attention_feature import Attention
from fold5_dataMIL import get_loader
import pandas as pd
import os
from PIL import Image
import numpy
import torchvision
from torchvision import datasets, transforms
import random
import torch, gc
# torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
from torch.utils.data import Dataset, DataLoader
import evaluation_test
from torch.optim import lr_scheduler
import time

parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument("--zoom", help='zoom_level',default=400)
parser.add_argument('--epochs',type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')

args = parser.parse_args()
cuda_gpu = torch.cuda.is_available()

writer = SummaryWriter(logdir="/log")

torch.manual_seed(args.seed)
if (cuda_gpu):
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')
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

img_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomResizedCrop((img_size)),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 这一组值是从imagenet训练集中抽样算出来的
        # 每张样本图像经过normalize后变成了均值为0 方差为1 的标准正态分布
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class CocoDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):  # 把传入的参数变为类中的参数，类的方法都可以调用
        all_labels = os.listdir(label_file)
        self.imgs = all_labels
        self.label_file = label_file
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):  # idex是随机取得16个batch
        # 间隔40取一张图片
        one_img = self.imgs[idx]
        self.labels = pd.read_csv(self.label_file + "/" + one_img, header=None)
        m = len(self.labels)
        random.seed(1)
        select_list = range(0, m)
        rand = random.sample(select_list, 20)
        patches = []
        labels = []
        label = self.labels.iloc[0, 1:].values
        label = label.astype('float32')
        for i in rand:
            patch = os.path.join(self.image_dir, self.labels.iloc[i, 0])
            pa = Image.open(patch)
            patch_img1 = np.array(pa)
            patch_img2 = self.transform(patch_img1)
            patches.append(patch_img2)
            labels.append(label)
        array = tuple(patches)
        array = torch.stack(array, 0)
        return array, label

class testDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):  # 把传入的参数变为类中的参数，类的方法都可以调用
        all_labels = os.listdir(label_file)
        self.imgs = all_labels
        self.label_file = label_file
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):  # idex是随机取得16个batch
        # 间隔40取一张图片
        one_img = self.imgs[idx]
        self.labels = pd.read_csv(self.label_file + "/" + one_img, header=None)
        m = len(self.labels)
        patches = []
        labels = []
        label = self.labels.iloc[0, 1:].values
        label = label.astype('float32')
        for i in range(m):
            patch = os.path.join(self.image_dir, self.labels.iloc[i, 0])
            pa = Image.open(patch)
            patch_img1 = np.array(pa)
            patch_img2 = self.transform(patch_img1)
            patches.append(patch_img2)
            labels.append(label)
        array = tuple(patches)
        array = torch.stack(array, 0)
        return array, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 将网络输入到cuda或者cpu中进行训练


# optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloaders, optimizer, epoch, critertion, scheduler, k):
    since = time.time()
    for phase in ['train', 'val']:
        loss_own = 0.
        running_loss = 0.
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
            pred_train = []
            ll_train = []
            # 依次获取所有图像，参与模型训练或测试
            for batch_idx, (data, label) in enumerate(dataloaders[phase]):
                bag_label = label
                data = data.type(torch.FloatTensor)
                if (cuda_gpu):
                    data, bag_label = data.cuda(), bag_label.cuda()
                data, bag_label = Variable(data), Variable(bag_label)   # 存放变化值的地理位置，tensor不能反向传播，variable可以反向传播
                # data = data.squeeze(0)   # 把第一维的图像的维数删除掉，现在剩下的第一维就是patch的数目

                optimizer.zero_grad()    # 把梯度置零
                # calculate loss and metrics
                Y_prob, Y_hat, A = model(data)
                # 更改的适合多标签分类的loss和accracy
                loss = critertion(Y_prob, bag_label)
                precision, recall, f1_score, pred = evaluation_test.calculate_acuracy_mode_one(Y_prob, bag_label)
                running_precision += precision
                running_recall += recall
                running_score += f1_score
                batch_num += 1
                # backward pass
                loss.backward()  # 反省传播计算得到每个参数的梯度值
                # step
                optimizer.step()  # 梯度下降执行一步参数更新
                loss_own += loss.item()

            epoch_loss = loss_own / len(dataloaders["train"])
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
                for batch_idx, (data, label) in enumerate(dataloaders[phase]):
                    # 获取输入
                    bag_label = label[0]
                    data = data.type(torch.FloatTensor)
                    if (cuda_gpu):
                        data, bag_label = data.cuda(), bag_label.cuda()
                    data, bag_label = Variable(data), Variable(bag_label)  # 存放变化值的地理位置，tensor不能反向传播，variable可以反向传播
                    data = data.squeeze(0)

                    # 网络前向运行
                    Y_prob, Y_hat, A = model(data)
                    # 计算Loss值
                    # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                    loss = critertion(Y_prob, bag_label)
                    precision, recall, f1_score, pred = evaluation_test.calculate_acuracy_mode_one(Y_prob,
                                                                                                   bag_label)
                    running_precision += precision
                    running_recall += recall
                    running_score += f1_score
                    batch_num += 1
                    loss_own += loss.item()

                epoch_loss = loss_own / len(dataloaders["val"])
                print('{} Loss: {:.4f} '.format("val", epoch_loss))
                epoch_precision = running_precision / batch_num
                print('{} Precision: {:.4f} '.format("val", epoch_precision))
                epoch_recall = running_recall / batch_num
                print('{} Recall: {:.4f} '.format("val", epoch_recall))
                epoch_score = running_score / batch_num
                print('{} f1_score: {:.4f} '.format("val", epoch_score))

                writer.add_scalar('val/val_loss', epoch_loss, epoch)
                writer.add_scalar('val/val_presicion', epoch_precision, epoch)
                writer.add_scalar('val/val_recall', epoch_recall, epoch)
                writer.add_scalar("val/val_score", epoch_score, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_score

if __name__ == "__main__":
    # img_save_dir = './AMIL_visualization/zoom_{}/epoch_{}'.format(zoom_level_x,epoch)
    # main_dir = "./" + zoom_level_x +'/'
    main_dir = "/model"

    n_fold = 5
    for k in range(n_fold):
        dataloaders, dataloaders_test = get_loader(k)
        model = Attention()
        if (cuda_gpu):
            model.cuda()
            print("model to gpu", k, "fold")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                               weight_decay=args.reg)
        scheduler = lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)

        criterion = BCEFocalLoss()

        # for epoch in range(1, args.epochs + 1):
        print('----------Start Training----------')
        epochs = 30
        best_score = 0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            val_acc = train(model, dataloaders, optimizer, epoch, criterion, scheduler, k)
            if val_acc > best_score:
                best_score = val_acc
                best_model = model
                torch.save(best_model, root_dir + "/" + str(k) + "_" + "attention_best.pt")
                print("the best model is in ", epoch)

        torch.save(model, main_dir + "/" + str(k) + "_" + "Attention_based_model.pt")
