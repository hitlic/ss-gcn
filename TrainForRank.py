from torchility import Trainer
from pytorch_lightning import LightningDataModule
import torch
from TaskModule import *

class TrainerForRank(Trainer):
    def compile(self, model: torch.nn.Module, loss, optimizer, data_module: LightningDataModule = None,
                log_loss_step=None, log_loss_epoch=True, metrics=None):
        self.task_module = TaskModule(model, loss, optimizer, log_loss_step, log_loss_epoch, metrics)
        self.data_module = data_module