import torch
import random as rand
from torch.utils.data import DataLoader
from torchility import Trainer
from torchility.callbacks import PrintProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import time
from torchility.callbacks import ModelAnalyzer
from torchility.callbacks import LRFinder
import matplotlib.pyplot as plt
from data import *
from model import *
from TaskModule import *
from TrainForRank import *
from metrics import *
from torchility.callbacks import ClassifierInterpreter


rand.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
else:
    torch.manual_seed(1)

#输入的只有特征矩阵X

# pygdata = load_pygdata('./datasets/wiki',head = True)
pygdata = load_pygdata('./datasets/co-auth',head = True)
# pygdata = load_pygdata('./datasets/cora',head = True)
# pygdata = load_pygdata('./datasets/metro',head = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = LearnToRankModel(feature_dim = 1, gcn_dims=[16,4], dense_dim = 16,ssl_dense_dim = 256)
#wiki
# model = LearnToRankModel(feature_dim = len(pygdata.x[0]), gcn_dims=[512,128], dense_dim = 16,ssl_dense_dim = 256)
#地铁512-512--------2000
model = LearnToRankModel(feature_dim = len(pygdata.x[0]), gcn_dims=[32,32,32], dense_dim = 16,ssl_dense_dim = 256)

def _collate_fn(batch):
    return batch[0]

train_dl = DataLoader([pygdata], batch_size = 1,collate_fn = _collate_fn)
val_dl = DataLoader([pygdata], batch_size = 1,collate_fn = _collate_fn)
test_dl = DataLoader([pygdata], batch_size = 1,collate_fn = _collate_fn)

#优化器
opt = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = TrainerForRank()
# compile
trainer.compile(model, rank_loss(),opt,metrics=[masked_MAP,masked_NDCG_at_n])
# train and validate 600
trainer.fit(train_dl, val_dl, 120)
# test
trainer.test(test_dl)




