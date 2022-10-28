"""
Define pytorch dataloader for DeepAttnMISL


"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os

from sklearn.model_selection import train_test_split

class MIL_dataloader():
    def __init__(self, train_path, val_path, test_path, cluster_num=7, train=True):

        if train:

            traindataset = MIL_dataset(list_path=train_path, cluster_num = cluster_num, train=train,
                              transform=transforms.Compose([ToTensor()]))

            traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=4)

            valdataset = MIL_dataset(list_path=val_path, train=False, cluster_num=cluster_num,
                                       transform=transforms.Compose([ToTensor()]))

            valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = [traindataloader, valdataloader]

        else:
            testdataset = MIL_dataset(list_path=test_path, cluster_num = cluster_num, train=False,
                              transform=transforms.Compose([ToTensor()]))
            testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = testloader

    def get_loader(self):
        return self.dataloader


class MIL_dataset(Dataset):
    def __init__(self, list_path, cluster_num,  transform=None, train=True):
        """
        Give npz file path
        :param list_path:
        """

        self.list_path = os.listdir(list_path)
        self.rootdir = list_path
        self.random = train
        self.transform = transform
        self.cluster_num = cluster_num

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):

        img_path = self.list_path[idx]

        Batch_set = []
        status_train = []

        all_vgg = []

        vgg_clus = [[] for i in range(self.cluster_num)]

        img_path = self.rootdir + "/" + img_path
        Train_vgg_file = np.load(img_path)  # 读取npy文件

        con_vgg, con_path, con_cluster = [], [], []

        mask = np.ones(self.cluster_num, dtype=np.float32)

        for i in range(1):  # How many wsi in the patient

            # np.savez(npz_file, feature=feature, cluster_name=cluster_name, label=label_img, patch_way=patches_path, img_ID=images_ID)
            cur_vgg = Train_vgg_file['feature']
            # cur_patient = Train_vgg_file['img_ID']
            cur_label = Train_vgg_file['label']
            # cur_time = Train_vgg_file['time']
            # cur_status = Train_vgg_file['status']
            cur_path = Train_vgg_file['patch_way']
            cur_cluster = Train_vgg_file['cluster_name']

            for id, each_patch_cls in enumerate(cur_cluster):  # 将不同类簇的特征分别放在不同的列表当中
                    vgg_clus[each_patch_cls].append(cur_vgg[id])

            Batch_set.append((cur_vgg, cur_cluster))

            np_vgg_fea = []
            for i in range(self.cluster_num):
                if len(vgg_clus[i]) == 0:
                    clus_feat = np.zeros((1, 2048), dtype=np.float32)
                    mask[i] = 0
                else:
                    if self.random:
                        curr_feat = vgg_clus[i]
                        ind = np.arange(len(curr_feat))  # 每一类簇内patch的数量n，然后从[0 1 2...n ]
                        np.random.shuffle(ind)  # 打断每一个类簇中patch的排序
                        clus_feat = np.asarray([curr_feat[i] for i in ind])
                    else:
                        clus_feat = np.asarray(vgg_clus[i])
                clus_feat = np.swapaxes(clus_feat, 1, 0)  # 交换两个轴的数据，第0维和第1维数据交换
                # clus_feat = np.expand_dims(clus_feat, 0)
                clus_feat = np.expand_dims(clus_feat, 1)  # 在axis=1中添加维度，索引clus_feat[1][0]后面的0不可更改
                np_vgg_fea.append(clus_feat)

            all_vgg.append(np_vgg_fea)

        for each_set in Batch_set:
            status_train.append(each_set[1])

        sample = {'feat': all_vgg[0], 'mask':mask, 'label':cur_label[0], 'cluster_num': cur_cluster, 'patch_way': cur_path[0]}
        # 类簇中元素为零时，对应位置的mask为零，其他位置为1

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cluster_num = 7
        image, label, mask = sample['feat'], sample['label'], sample['mask']

        return {'feat': [torch.from_numpy(image[i]) for i in range(cluster_num)], 'label': torch.FloatTensor([label]),
                'mask': torch.from_numpy(sample['mask']),
                'cluster_num': torch.from_numpy(sample['cluster_num']),
                'patch_way': sample['patch_way']
                }