import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.nn.init as init

class textCNN(nn.Module):
    def __init__(self, param):
        super(textCNN, self).__init__()
        kernel_num = param['kernel_num'] # output chanel size 16
        kernel_size = param['kernel_size']  # 3，4，5
        vocab_size = param['vocab_size']  # 22906
        embed_dim = param['embed_dim']  # 60
        dropout = param['dropout']  # 0.5
        class_num = param['class_num']  # 5
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)  #  Embedding(22906, 60, padding_idx=1)  # class Embedding(num_embeddings词嵌入词典大小: int, embedding_dim: int
        self.conv11 = nn.Conv1d(embed_dim, kernel_num, kernel_size[0])  #   # class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv12 = nn.Conv1d(embed_dim, kernel_num, kernel_size[1])  #   # 卷积核高度与词向量维度一致，为60
        self.conv13 = nn.Conv1d(embed_dim, kernel_num, kernel_size[2])  #  
        self.dropout = nn.Dropout(dropout)  # Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)  #  Linear(in_features=48, out_features=5, bias=True)  将（128，48）==》（128，5）    # 对输入数据做线性变换：y=Ax+b  
        # 参数：
        # in_features - 每个输入样本的大小
        # out_features - 每个输出样本的大小
        # bias - 若设置为False，这层不会学习偏置。默认值：True

        # 形状：
        # 输入: (N,in_features)
        # 输出： (N,out_features)

        # 变量：
        # weight -形状为(out_features x in_features)的模块中可学习的权值
        # bias -形状为(out_features)的模块中可学习的偏置

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        #  torch.Size([128, 60, 20])
        x = conv(x)
        # 经过一维卷积后的大小 torch.Size([128, 16, 18])  
        x = F.relu(x) 
        # 激活层后：torch.Size([128, 16, 18])
        x = F.max_pool1d(x, x.size(2))  #(128,16,1)  # torch.nn.functional.max_pool1d(input([128, 16, 18]), kernel_size 18)     # x.size(2)指H_out的值
        x = x.squeeze(2)
        #  (batch, kernel_num)   torch.Size([128, 16])         (128,16,1) .squeeze(2)==> (128,16)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)  (128, 20)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)   经过embedding层后(128, 20, 60)
        # TODO init embed matrix with pre-trained
        x = x.permute(0,2,1)  # 将(128, 20, 60)换为(128, 60, 20)

        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)  torch.Size([128, 16])
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)  torch.Size([128, 16])
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num) (128,3*16) = (128,48)   torch.cat(inputs, dimension=0) → Tensor
        x = self.dropout(x)  # torch.Size([128, 48])
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()