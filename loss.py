import torch
import torch.nn as nn
import torch.nn.functional as F


class rank_loss(nn.Module):
    
    def __init__(self):
        super().__init__()

    
    def forward(self,pred,target,mask):
        mse_for_regression = F.mse_loss(pred[0][mask], target[0][mask])
        top_one_loss = self.masked_TOP_loss(pred[0], target[0],mask)
        # mlp的时候注掉
        mse_for_ssl = F.mse_loss(pred[1],target[1])
        #原始任务
        # return 0.3 * mse_for_regression + top_one_loss + 0.8 * mse_for_ssl
        #去除自监督任务
        # return 0.3 * mse_for_regression + top_one_loss + 0.0 * mse_for_ssl
        #去除回归任务
        return 0.0 * mse_for_regression + top_one_loss + 0.8 * mse_for_ssl
        #去除排序任务
        # return 0.3 * mse_for_regression + 0 * top_one_loss + 0.8 * mse_for_ssl
        #只有排序任务
        # return 0.0 * mse_for_regression + top_one_loss + 0.0 * mse_for_ssl
        #只有回归任务
        # return 0.3 * mse_for_regression + 0.0 * top_one_loss + 0.0 * mse_for_ssl
        # node2vec
        # return 0.3 * mse_for_regression + top_one_loss

    def masked_TOP_loss(self,pred, target,mask):
        """
        Top One Probability(TOP) loss，from <<Learning to Rank: From Pairwise Approach to Listwise Approach>>
        """
        # pred = torch.squeeze(pred[target.mask])
        # target = target.data[target.mask]

        pred = torch.squeeze(pred[mask])
        # target = target[mask]
        target = torch.squeeze(target.data[mask])
        # pred_p = torch.exp(pred)
        # target_p = torch.exp(target)
        pred_p = torch.softmax(pred, 0)
        target_p = torch.softmax(target, 0)
        loss = torch.mean(-torch.sum(target_p*torch.log(pred_p+1e-9)))
        return loss

if __name__ == '__main__':
    x_pre = [0.2,0.3,0.4,0.2,0.6,0.8,0.3,0.6,0.1,0.9]
    y_target = [11,12,2,4,1,7,8,3,9,2]
    x=torch.zeros(10) #返回一个全为标量0的张量，形状可由可变参数*size定义 stock_lenth-sqe_len行，5列
    y=torch.zeros(10)
    mask = [True,True,False,False,False,False,True,False,False,False]
