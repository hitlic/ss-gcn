import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import warnings
from data import *
import random as rand
import networkx as nx
from metrics import *
from loss import *
warnings.filterwarnings('ignore')  # 屏蔽警告信息

# 保证每次结果都一样，排除随机因素
np.random.seed(1)
torch.random.manual_seed(1.0)

# 1. ---- 数据

class embData(pl.LightningDataModule):
    def __init__(self, batch_size=5213):
        super().__init__()
        # self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):     # 加载、准备数据
        # coauth
        embedding_result = open("./datasets/co-auth/embeddings.emb")
        # embedding_result = open("./datasets/metro/embeddings.emb")
        # cora
        # embedding_result = open("./datasets/cora/embeddings_cora.emb")
        #wiki数据集
        # embedding_result = open("./datasets/wiki/embeddings.emb")
        emb_list = []
        num_dic = {}
        for i,line in enumerate(embedding_result):
            if i == 0:
                continue
            else:
                num_dic[i-1] = line.split(' ', 1)[0].strip()
                emb = [float(val) for val in line.split(' ', 1)[1].strip().split(' ')]
                emb_list.append(emb)
        #划分数据集
        # coauth
        g = load_graph("./datasets/co-auth", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
        # cora
        # g = load_graph("./datasets/cora", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
        # wiki
        # g = load_graph("./datasets/wiki", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
        # 地铁
        # g = load_graph("./datasets/metro", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
        label_dict = nx.get_node_attributes(g, "__label__")
        label_list = []
        for i,emb in enumerate(emb_list):
            node_num = num_dic[i]
            label_list.append(label_dict[node_num])
         
        train_per = 0.6
        val_per = 0.2
        train_mask = [False] * g.number_of_nodes()
        val_mask = [False] * g.number_of_nodes()
        test_mask = [False] * g.number_of_nodes()
        a = b = c = 0
        for i,node in enumerate(g.nodes()):
            p = rand.random() 
            if p < train_per:
                if label_list[i][0] == -1.0:
                    train_mask[i] = False
                else:
                    train_mask[i] = True
                    a=a+1
            elif train_per <= p < train_per+val_per:
                if label_list[i][0] == -1.0:
                    val_mask[i] = False
                else:
                    val_mask[i] = True
                    b=b+1
            else:
                if label_list[i][0] == -1.0:
                    test_mask[i] = False
                else:
                    test_mask[i] = True
                    c=c+1
        print(a,"!!!   ",b,"!!   ",c)

        self.train_data = torch.from_numpy(np.array(emb_list)).float() 
        self.test_data = torch.from_numpy(np.array(emb_list)).float() 
        self.val_data = torch.from_numpy(np.array(emb_list)).float() 
        self.train_label = torch.from_numpy(np.array(label_list).reshape(-1)).float() 
        self.test_label = torch.from_numpy(np.array(label_list).reshape(-1)).float() 
        self.val_label = torch.from_numpy(np.array(label_list).reshape(-1)).float()
        self.train_mask =  torch.from_numpy(np.array(train_mask).reshape(-1))
        self.val_mask =  torch.from_numpy(np.array(val_mask).reshape(-1))
        self.test_mask =  torch.from_numpy(np.array(test_mask).reshape(-1))


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:      # 训练和验证数据配置
            self.train_ds = list(zip(self.train_data, self.train_label,self.val_mask))
            self.valid_ds = list(zip(self.val_data, self.val_label,self.val_mask))
        if stage == 'test' or stage is None:     # 测试数据配置
            self.test_ds = list(zip(self.test_data, self.test_label,self.test_mask))
        return super().setup(stage)

    def train_dataloader(self): # 训练 data loader
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   # 验证 data loader
        return DataLoader(self.valid_ds, batch_size=self.batch_size)

    def test_dataloader(self):  # 测试 data loader
        return DataLoader(self.test_ds, batch_size=self.batch_size)


# 2. --- 模型
class MultiLayersModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)
        )

    def forward(self, x): 
        out = self.model(x)                      # 前向计算
        return out                      # 前向计算


    def get_loss(self, pred, target,mask):
        my_loss = rank_loss()
        loss = my_loss(pred.squeeze().unsqueeze(0), target.unsqueeze(0),mask)
        # mse_for_regression = F.mse_loss(pred[mask], target[mask])
        return loss

    def get_acc(self, logits, label,mask):
        preds = torch.argmax(logits, dim=1)
        map_val = masked_MAP(logits.squeeze().unsqueeze(0), label.unsqueeze(0), mask)
        map_dict = {"map_"+key : value for key,value in map_val.items()}
        ndcg = masked_NDCG_at_n(logits.squeeze().unsqueeze(0), label.unsqueeze(0), mask)
        ndcg_dict = {"ndcg_"+key : value for key,value in ndcg.items()}
        kendell_val = kendall(logits.squeeze().unsqueeze(0), label.unsqueeze(0), mask)
        result_dic = {**map_dict, **ndcg_dict,**kendell_val} 
        # acc = accuracy(preds, torch.argmax(label, 1))
       
        return result_dic

    def training_step(self, batch, batch_nb):   # 训练步
        x, y, mask = batch
        logits = self(x)
        loss = self.get_loss(logits, y, mask)
        acc = self.get_acc(logits, y, mask)
        self.log('train_loss', loss,prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb): # 验证步
        x, y, mask = batch
        logits = self(x)
        loss = self.get_loss(logits, y, mask)
        acc = self.get_acc(logits, y, mask)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_nb):       # 测试步
        x, y, mask = batch
        logits = self(x)
        loss = self.get_loss(logits, y, mask)
        acc = self.get_acc(logits, y, mask)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):             # 优化器
        return torch.optim.Adam(self.parameters(), lr=0.001)


# 3. --- 训练
dm = embData(6000)
model = MultiLayersModel()
trainer = pl.Trainer(max_epochs=200)
trainer.fit(model, dm)
result = trainer.test(model, dm)
print(result)
