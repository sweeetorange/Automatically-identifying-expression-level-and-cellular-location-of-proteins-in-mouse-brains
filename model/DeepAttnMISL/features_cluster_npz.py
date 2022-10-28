import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from scipy.cluster.vq import *
import matplotlib .pyplot as plt
from pylab import *
from scipy import *
from imutils import paths
import pandas as pd
import numpy as np
import random

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.net = torchvision.models.resnet50(pretrained=True)
        self.resnet_layer = nn.Sequential(*list(self.net.children())[:-2])

        self.feature_extractor = nn.Sequential()

    def forward(self, x):
        h = self.resnet_layer(x)


if __name__ == '__main__':
    # 预处理操作
    to_tensor = transforms.Compose([transforms.ToPILImage(),
        # transforms.RandomResizedCrop((img_size)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    # print(list(model.children()))
    features = list(model.children())[:-1]  # 去掉池化层及全连接层
    # print(list(model.children())[:-1])
    modelout = nn.Sequential(*features).to(device)

    k = 7
    # data
    # image_file = "D:/360Downloads/code/DeepAttnMISL_MEDIA-master/data"
    images_file = "data/MIL_data/val"
    csv_files = "data/all_csvs/val"
    # path_all = sorted(list(paths.list_images(image_file)))
    imgs = os.listdir(csv_files)
    image_names = []

    names = []
    for m in range(len(imgs)):
        labels = []
        img_name = imgs[m]
        name = img_name.split(".")[0]
        names.append(name)
        out_npz = "/npz_val/" + name + ".npz"
        csv_re = pd.read_csv(csv_files + "/" + img_name, header=None)
        one_label = csv_re.iloc[0, 1:].values.astype("float32")
        patch_name = csv_re.iloc[0, 0]
        label_fi = patch_name.split("/")[0]
        img_fi = patch_name.split("/")[1]
        patches_fi = images_file + "/" + label_fi + "/" + img_fi
        all_patches = os.listdir(patches_fi)
        l = len(all_patches)
        # print(l)
        select_patches = []
        if l < 6:
            continue
        if l > 128:
            random.seed(1)
            select_list = range(0, l)
            rand = random.sample(select_list, 128)
            for n in rand:
                select_patches.append(all_patches[n])
            all_patches = select_patches
        imgs_read = []
        patches_path = []
        for i in range(len(all_patches)):
            patch_way = patches_fi + "/" + all_patches[i]
            one_patch_path = label_fi + "/" + img_fi + "/" + all_patches[i]
            labels.append(one_label)
            patches_path.append(one_patch_path)
            img = Image.open(patch_way)
            img1 = np.array(img)
            img_tensor = to_tensor(img1).unsqueeze(0).to(device, torch.float)
            imgs_read.append(img_tensor)

        imgs_read1 = torch.tensor([item.cpu().detach().numpy() for item in imgs_read]).cuda()
        # imgs_read1 = torch.tensor([item.cpu().detach().numpy() for item in imgs_read1]).cuda()
        imgs_read2 = imgs_read1.squeeze()
        # print(imgs_read2.size(), name)
        out = modelout(imgs_read2)
        out = out.squeeze()

        # K_means
        feature = out.cpu().detach().numpy()
        # feature = whiten(feature)  # 对输入数据按标准差做归一化
        codebook, distortion = kmeans(feature, k)
        code, distance = vq(feature, codebook)

        cluster_name = np.array(code)
        patches_path = np.array(patches_path)
        images_ID = np.array(names)
        label_imgs = np.array(labels)
        # save npy file
        npz_file = out_npz
        np.savez(npz_file, feature=feature, cluster_name=cluster_name, label=label_imgs, patch_way=patches_path, img_ID=images_ID)
        torch.cuda.empty_cache()



