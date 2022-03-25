from torchility import tasks
# from .metrics import MetricBase
from metrics import *
class TaskModule(tasks.GeneralTaskModule):

    def forward(self, batch_data):                          # 前向计算
        return self.model(batch_data)

    def training_step(self, batch, batch_nb):               # 训练步
        loss, preds, targets = self.do_forward(batch,batch.train_mask)
        # self.log('train_loss',loss,on_step=self.on_step,on_epoch=self.on_epoch)
        self.log('train_loss',loss,on_step=True,on_epoch=True)
        self.do_metric(preds, targets, batch.val_mask, self.log)
        self.messages['train_batch'] = (batch_nb, preds, targets) # (batch_idx, preds, tagets)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, preds, targets = self.do_forward(batch,batch.val_mask) # 验证步
        # self.log('val_loss', loss, prog_bar=True, on_step=self.on_step, on_epoch=self.on_epoch)
        self.log('val_loss', loss, prog_bar=True,on_step=True,on_epoch=True)
        self.do_metric(preds, targets, batch.val_mask, self.log)
        self.messages['val_batch'] = (batch_nb, preds, targets) # (batch_idx, preds, tagets)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):                   # 测试步
        loss, preds, targets = self.do_forward(batch,batch.test_mask)
        self.log('test_loss', loss, prog_bar=True)
        self.do_metric(preds, targets,batch.test_mask, self.log)
        self.messages['test_batch'] = (batch_nb, preds, targets) # (batch_idx, preds, tagets)
        return {'test_loss': loss}

    def configure_optimizers(self):                         # 优化器
        return self.opt

    def do_forward(self, batch,mask):                          # 前向计算
        input_feat, targets = (batch.x,batch.edge_index),(batch.y,batch.y_dis)
        preds = self(input_feat)
        loss = self.loss_fn(preds, targets,mask)
        return loss, preds, targets

    def do_metric(self, preds, targets,mask, log):               # 指标计算
        self.metrics = [kendall,masked_MAP,masked_NDCG_at_n]
        for metric in self.metrics:
            result = metric(preds, targets,mask)
            # if isinstance(metric, MetricBase):
            #     name = metric.name
            #     on_step = metric.log_step
            #     on_epoch = metric.log_epoch
            # else:
            name = metric.__name__
            on_step, on_epoch = True, True
            self.log(name,result, prog_bar=True, on_step=on_step, on_epoch=on_epoch)

