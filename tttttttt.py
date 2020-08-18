import torch
import torch.nn as nn
import torch.nn.functional as F

x1 = torch.Tensor(128,1,20,60)
conv2d = nn.Conv2d(1,16,(3,60))
print(conv2d(x1).size())  # torch.Size([128, 16, 18, 1])

x2 = torch.Tensor(128,60,20)  # input = torch.randn(64,1,8)#batch_size =64,1就对应词向量维度，8=句子长度
conv1d = nn.Conv1d(60,16,3)
print(conv1d(x2).size())  #torch.Size([128, 16, 18])

x3 = conv1d(x2) # 128,16,18
x4 = F.max_pool1d(x3, x3.size(2))
print(x4.size()) # 128,16,1

x5 = F.max_pool2d(x3, x3.size(2))
print(x5.size()) # 128,16,1