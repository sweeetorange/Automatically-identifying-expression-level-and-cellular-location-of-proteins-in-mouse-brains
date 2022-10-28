from __future__ import print_function

import numpy as np
import imageio
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from deepattnmisl_model import DeepAttnMIL_Surv
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
from tqdm import tqdm
from load_dataset import MIL_dataloader

parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument("--zoom", help='zoom_level',default=400)
parser.add_argument('--epochs',type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')

args = parser.parse_args()
cuda_gpu = torch.cuda.is_available()

writer = SummaryWriter(logdir="/log")
# writer = SummaryWriter(logdir="D:/360Downloads/code/DeepAttnMISL_MEDIA-master/own_code/log")

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



# optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, traindata, valdata, optimizer, epoch, critertion, k, cluster_num):
    since = time.time()
    sigmoid_fun = nn.Sigmoid()
    for phase in ['train', 'val']:
        loss_own = 0.
        running_loss = 0.
        running_precision = 0.
        running_recall = 0.
        running_score = 0.
        batch_num = 0
        if phase == 'train':
            #  调用模型训练
            model.train()
            pred_train = []
            ll_train = []
            # 依次获取所有图像，参与模型训练或测试
            tbar = tqdm(traindata, desc='\r')  # 是一个快速可扩展的进度条
            for i_batch, sampled_batch in enumerate(tbar):
                X, label, mask, patch_path = sampled_batch['feat'], sampled_batch['label'], sampled_batch['mask'],sampled_batch['patch_way']
                # X为特征，lbl为label，mask为类簇的矩阵，使用mask的目的是：当某一类没有patch时，训练时不需要为这类簇分配权重

                graph = [X[i].cuda() for i in range(cluster_num)]
                label = label.cuda()
                masked_cls = mask.cuda()

                # ===================forward=====================
                optimizer.zero_grad()    # 把梯度置零
                # calculate loss and metrics

                Y_prob = model(graph, masked_cls)  # prediction
                # print("prediction is:", Y_prob)
                # 更改的适合多标签分类的loss和accracy
                # print("prediction:", Y_prob, "sigmoid", sigmoid_fun(Y_prob), label)
                loss = critertion(sigmoid_fun(Y_prob), label)
                # print("sigmoid is:", sigmoid_fun(Y_prob))
                precision, recall, f1_score, pred = evaluation_test.calculate_acuracy_mode_one(sigmoid_fun(Y_prob), label)
                # print(precision, recall, f1_score, pred )
                running_precision += precision
                running_recall += recall
                running_score += f1_score
                batch_num += 1
                # backward pass
                loss.backward()  # 反省传播计算得到每个参数的梯度值
                # step
                optimizer.step()  # 梯度下降执行一步参数更新
                loss_own += loss.item()

            epoch_loss = loss_own / len(trainloader)
            print(len(trainloader))
            print('{} Loss: {:.4f} '.format("train", epoch_loss))
            epoch_precision = running_precision / batch_num
            print('{} Precision: {:.4f} '.format("train", epoch_precision))
            epoch_recall = running_recall / batch_num
            print('{} Recall: {:.4f} '.format("train", epoch_recall))
            epoch_score = running_score / batch_num
            print('{} f1_score: {:.4f} '.format("train", epoch_score))

            writer.add_scalar('train/train_loss', epoch_loss, epoch + (k-1) * 200)
            writer.add_scalar('train/train_presicion', epoch_precision, epoch + (k-1) * 200)
            writer.add_scalar('train/train_recall', epoch_recall, epoch + (k-1) * 200)
            writer.add_scalar('train/train_score', epoch_score, epoch + (k-1) * 200)

        else:
            with torch.no_grad():
                # 依次获取所有图像，参与模型训练或测试
                model.eval()
                pred_val = []
                ll_val = []
                tbar = tqdm(valdata, desc='\r')  # 是一个快速可扩展的进度条
                for i_batch, sampled_batch in enumerate(tbar):
                    X, label, mask, patch_path = sampled_batch['feat'], sampled_batch['label'], sampled_batch['mask'], \
                                                 sampled_batch['patch_way']
                    # X为特征，lbl为label，mask为类簇的矩阵，使用mask的目的是：当某一类没有patch时，训练时不需要为这类簇分配权重
                    graph = [X[i].cuda() for i in range(cluster_num)]
                    label = label.cuda()
                    masked_cls = mask.cuda()
                    # calculate loss and metrics
                    Y_prob = model(graph, masked_cls)

                    # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                    loss = critertion(sigmoid_fun(Y_prob), label)
                    precision, recall, f1_score, pred = evaluation_test.calculate_acuracy_mode_one(sigmoid_fun(Y_prob), label)

                    running_precision += precision
                    running_recall += recall
                    running_score += f1_score
                    batch_num += 1
                    loss_own += loss.item()

                epoch_loss = loss_own / len(valloader)
                print('{} Loss: {:.4f} '.format("val", epoch_loss))
                epoch_precision = running_precision / batch_num
                print('{} Precision: {:.4f} '.format("val", epoch_precision))
                epoch_recall = running_recall / batch_num
                print('{} Recall: {:.4f} '.format("val", epoch_recall))
                epoch_scoreval = running_score / batch_num
                print('{} f1_score: {:.4f} '.format("val", epoch_scoreval))

                writer.add_scalar('val/val_loss', epoch_loss, epoch + (k-1) * 200)
                writer.add_scalar('val/val_presicion', epoch_precision, epoch + (k-1) * 200)
                writer.add_scalar('val/val_recall', epoch_recall, epoch + (k-1) * 200)
                writer.add_scalar("val/val_score", epoch_scoreval, epoch + (k-1) * 200)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_scoreval


if __name__ == "__main__":
    # img_save_dir = './AMIL_visualization/zoom_{}/epoch_{}'.format(zoom_level_x,epoch)
    # main_dir = "./" + zoom_level_x +'/'
    main_dir = "/model"

    print('Init Model')

    # for epoch in range(1, args.epochs + 1):
    print('----------Start Training----------')
    epochs = 200

    clus = 7
    n_fold = 6
    for k in range(1, n_fold):

        train_path = "data/npz_train/" + str(k) + "fold"
        val_path = "data/npz_val/" + str(k) + "fold"
        test_path = "data/npz_test/1fold"

        train_datasets = MIL_dataloader(train_path=train_path, val_path=val_path, test_path=test_path, cluster_num=clus,
                                        train=True)
        trainloader, valloader = train_datasets.get_loader()

        TestData = MIL_dataloader(train_path=train_path, val_path=val_path, test_path=test_path, cluster_num=clus, train=False)
        testloader = TestData.get_loader()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 将网络输入到cuda或者cpu中进行训练
        model = DeepAttnMIL_Surv(cluster_num=clus)
        if (cuda_gpu):
            model.cuda()
            print("model to gpu")
            print(str(k) + "fold")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = BCEFocalLoss()

        max_val = 0
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            val_acc = train(model, trainloader, valloader, optimizer, epoch, criterion, k, cluster_num=clus)
            if val_acc > max_val:
                max_val = val_acc
                torch.save(model, main_dir + "/" + str(k) + "bestmodel_cluster7")

            test_acc = test(model, testloader, criterion, epoch, k, cluster=clus)

        torch.save(model ,main_dir+"/"+ str(k)+ "deepattnmisl.pt")
