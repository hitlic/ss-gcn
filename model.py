import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from loss import *
from metrics import *
import numpy as np
from torchility import tasks


#模型
# 2. --- 模型
class LearnToRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dims=(32, 32), dense_dim=64,ssl_dense_dim=64):
        super().__init__()
        #把神经网络画出来，我这里就两层GCN加两层全连接
        self.gcns = nn.ModuleList()
        in_dim = feature_dim
        for dim in gcn_dims:
            self.gcns.append(GCNRelu(in_dim, dim))
            in_dim = dim
        self.dense = MLP(gcn_dims[-1], [dense_dim])
        # self.ssl_dense = MLP(gcn_dims[-1], [ssl_dense_dim])
        self.ssl_out = nn.Linear(gcn_dims[-1], 3)
        self.out = nn.Linear(dense_dim, 1)


 #前向传播
    def forward(self, data):
        x, edge_index = data[0], data[1]
        for gcn in self.gcns:
            x = gcn(x, edge_index)
            # gcn.register_forward_hook(farward_hook)
            # gcn.register_backward_hook(backward_hook)
        outputs = self.dense(x)
        #ssl_outputs = self.ssl_dense(x)
        ssl_outputs = self.ssl_out(x)
        outputs = self.out(outputs)
        return (outputs,ssl_outputs)


#定义卷积层，激活函数是leakyRelu
class GCNRelu(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.relu = GeneralRelu(leak=0.1, sub=0.4, maxv=6.)
        #self.out = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index):
        outputs = self.gcn(x, edge_index)
        outputs = self.relu(outputs)
        #outputs = self.out(outputs)
        return outputs

#定义激活函数，如果leak, sub, maxv有传参，则使用leakyRelu，否则使用relu
class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x

#全连接层
class MLP(nn.Module):
    def __init__(self, in_dim, out_dims):
        super().__init__()
        self.linears = nn.Sequential()
        for i, dim in enumerate(out_dims):
            self.linears.add_module(f'linear_relu_{i}', LinearRelu(in_dim, dim))
            in_dim = dim
    def forward(self, x):
        return self.linears(x)


class LinearRelu(nn.Module):
    def __init__(self, ind_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(ind_dim, out_dim)
        self.relu = GeneralRelu(leak=0.1, sub=0.4, maxv=6.)

    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.relu(outputs)
        return outputs
