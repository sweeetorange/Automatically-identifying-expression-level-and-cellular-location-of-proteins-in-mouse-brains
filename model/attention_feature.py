import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torchvision

criterion = nn.MultiLabelSoftMarginLoss()

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()  # 这是对继承自父类的属性进行初始化，而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化
        self.L = 500
        self.D = 128
        self.K = 1

        net = torchvision.models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()  # 将分类层置空
        self.feature = net
        self.classifier = nn.Sequential(   # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 800),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(800, self.L),           # linear 是全连接层 L=500
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier_last = nn.Sequential(
            nn.Linear(self.L*self.K, 8),     # l=500, K=1
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('========================')
        # print(x.shape)
        x = x.squeeze(0)   # 主要对数据的维度进行压缩，去掉维数为1的维度
        # print(x.shape)
        # print('========================')
        H = self.feature(x)
        H = H.view(x.size(0), -1)
        H = self.classifier(H)
        H = H.view(-1, 800)          # 相当于resize
        H = self.feature_extractor_part2(H)  # NxL   none*500

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN   transpose交换A的两个梯度
        A = F.softmax(A)  # softmax over N

        M = torch.mm(A, H)  # KxL = 1*500

        Y_prob = self.classifier_last(M)
        Y_hat = torch.ge(Y_prob, 0.5).double()    # 比较两个值的大小

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        # torch.eq:对两个张量进行逐元素比较，相同位置的两个元素相同则为true，不同则为false
        # item()：将一个零维张量转换成浮点数

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)  # 将输入input张量的每个元素都转换到区间[min,max]中
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    def critertion_multilabel(self, data, bag_label):
        Y = bag_label.float()
        Y_prob, _, A = self.forward(data)
        loss = criterion(Y_prob, Y)
        return loss

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        net = torchvision.models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()  # 将分类层置空
        self.feature = net
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 800),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(800, self.L),  # linear 是全连接层 L=500
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)  # 主要对数据的维度进行压缩，去掉维数为1的维度
        # print(x.shape)
        # print('========================')
        H = self.feature(x)
        H = H.view(x.size(0), -1)
        H = self.classifier(H)
        H = H.view(-1, 800)  # 相当于resize
        H = self.feature_extractor_part2(H) # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.sigmoid(A)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A