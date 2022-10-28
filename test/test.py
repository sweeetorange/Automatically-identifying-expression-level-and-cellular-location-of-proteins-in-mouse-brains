import os
import numpy as np
import imageio
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import os
from PIL import Image
import numpy
import torchvision
from torchvision import datasets, transforms
import random
import torch, gc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import evaluation_test

img_size = 224
data_transforms = {
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class testDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):  
        all_labels = os.listdir(label_file)
        self.imgs = all_labels
        self.label_file = label_file
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):  
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



def test(model, dataloaders_test, criterion, m):
    print("test started")
    pred_test = []
    ll_test = []
    model.eval()
    test_loss = 0
    correct = 0
    running_precision = 0.
    running_recall = 0.
    running_score = 0.
    batch_num = 0.
    sigmoid_fun = nn.Sigmoid()
    with torch.no_grad():
        for data, target in dataloaders_test:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data = data.squeeze(0)
            output = model(data)
            v = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
            vote_result = torch.from_numpy(v)
            out = sigmoid_fun(output)
            a = out.size(0)
            
            for i in range(8):
                for j in range(a):
                    bb = out[j, i]
                    if bb > 0.5:
                        out[j, i] = 1
                    else:
                        out[j, i] = 0
            for k in range(8):
                c = out[:, k]
                cc = a * 0.5
                number = c.sum()
                if number > cc:
                    vote_result[0, k] = 1
                else:
                    vote_result[0, k] = 0
            vote_result = vote_result.float().cuda()
            loss = criterion(vote_result, target)  # sum up batch loss
            precision, recall, f1_score, pred = evaluation_test.calculate_acuracy_mode_one(vote_result, target)
            test_loss += loss.item()
            running_precision += precision
            running_recall += recall
            running_score += f1_score
            batch_num += 1

        epoch_loss = test_loss / len(dataloaders_test)
        epoch_precision = running_precision / batch_num
        epoch_recall = running_recall / batch_num
        epoch_score = running_score / batch_num
        result_test = 'loss:{:.4f}, recall: {:.4f}, precision: {:.2f}, f1_score: {:.4f}'.format(
            epoch_loss, epoch_recall, epoch_precision, epoch_score)
        print(result_test)


def main():
    since = time.time()

    image_datasets_test = testDataset("data/test/test_csv/",
                                      '/test_image/',
                                      data_transforms['test'])
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    n_fold = 5
    for k in range(n_fold):
        model_file = "/model/" + str(k) + "_" + "densenet.pt"
        model = torch.load(model_file)
        print("model compile")
        criterion = nn.BCELoss()
        test(model, dataloaders_test, criterion, k)

if __name__ == '__main__':
    main()
