import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

# --- metrics


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


def masked_accuracy(pred, target):
    _, pred = pred.max(dim=1)
    correct = pred[target.mask].eq(target.data[target.mask]).sum()
    acc = correct / target.mask.float().sum()
    return acc


def masked_mse(pred, target):
    return F.mse_loss(torch.squeeze(pred[target.mask]), target.data[target.mask])


def masked_mae(pred, target):
    pred = torch.squeeze(pred[target.mask])
    target = target.data[target.mask]
    return torch.mean(torch.abs(pred - target))


def ordinal(pred, target):
    """
    真实排序下，对应的预测值的序号
    """
    #.transpose() 转置,将预测值和真实值拼起来
    frame = np.array([pred, target]).transpose()
    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    #按第一列从大到小排列后输出
    frame = frame[(-frame[:, 0]).argsort()]
    #concatenate 数组拼接
    frame = np.concatenate([frame, np.expand_dims(np.arange(1, len(pred)+1), 1)], 1)
    frame = frame[(-frame[:, 1]).argsort()]
    pred_ids = frame[:, 2]
    return pred_ids


def MAP(pred, target):
    """
    Mean average precision(MAP)
    """
    n = len(pred)
    target_ids = np.arange(1, len(pred)+1)
    pred_ids = ordinal(pred, target)
    
    def p_at_n(p_ids, t_ids, n):
        #取交集的长度
        inter_len = len(set(p_ids[:n]).intersection(set(t_ids[:n])))
        p = inter_len/n
        return p
    #cora数据集
    # top_list = [10,20,50,100,len(pred)]
    # 合著网络数据集
    # top_list = [10,20,50,100,200,300,len(pred)]
    #wiki数据集
    # top_list = [10,20,50,100,200,300,len(pred)]
    # 地铁数据集
    top_list = [10,20,30,len(pred)]
    map_result_dict = {}
    
    # for j in top_list:
    #     p_list = []
    #     for i in range(1,j+1):
    #         p_list.append(p_at_n(pred_ids, target_ids, i))
    #     map_result_dict[str(j)] = np.average(np.array(p_list))
    p_list = []
    for i in range(1,len(pred)+1):
             p_list.append(p_at_n(pred_ids, target_ids, i))
    map_result_dict["MAP"] = np.average(np.array(p_list))
    return map_result_dict


def masked_MAP(pred, target,mask):
    """
    Mean average precision(MAP)
    """
    pred = torch.squeeze(pred[0][mask]).detach().cpu().numpy()
    target = target[0].data[mask].detach().cpu().numpy()
    target = target.reshape(len(target))
    # label_pred_np,label_target_np = get_label_data(pred,target)
    return MAP(pred, target)


def DCG_at_n(pred, target, n):
    """
    Discount Cumulative Gain (DCG@n)
    """
    top_list = n
    result_dict = {}
    for top in top_list:
        #transpose() 转置，开始结构[[预测值（全部）][真实值（全部）]],维度(2,56)，转置之后[[真实值,预测值][真实值,预测值]...],维度(56,2)真实值与预测值一一对应
        frame = np.array([pred, target]).transpose()
        # 取出所有的预测值(也就是第一维的数据),取负之后,从小到大排列之后取其索引[.argsort() 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y]
        frame = frame[(-frame[:, 0]).argsort()]
        frame = frame[:top]
        result = np.sum([t/np.log2(i+2) for i, (_, t) in enumerate(frame)])
        result_dict[top] = result
    return result_dict



def NDCG_at_n(n, pred, target):
    """
    Normalized discount cumulative gain (NDCG@n)
    """

    pred_dict = DCG_at_n(pred, target, n)
    perfect_dict = DCG_at_n(target, target, n)
    result_list = {}
    for val in n:
        result = pred_dict[val]/perfect_dict[val]
        result_list[str(val)] = result
    return result_list


def masked_NDCG_at_n(pred, target,mask):
    """
    Masked normalized discount cumulative gain (NDCG@n)
    """
    pred = torch.squeeze(pred[0][mask]).detach().cpu().numpy()
    target = target[0].data[mask].detach().cpu().numpy()
    target = target.reshape(len(target))
    #取出那些有标签的来算，不然也没意义
    # label_pred_np,label_target_np = get_label_data(pred,target)
    # cora数据集
    # top_list = [10,20,30,50,100,len(pred)]
    # 合著网络数据集
    top_list = [10,20,50,100,200,300,len(pred)]
    #wiki数据集
    # top_list = [10,20,50,100,200,300,len(pred)]
    # 地铁数据集
    # top_list = [10,15,20,25,30,35,40,len(pred)]
    result_dict = {}
    list_target = np.zeros(len(target), dtype=int)
    for i,val in enumerate(target.argsort()):
        list_target[val] = i+1
    for top in top_list:
        result = ndcg_score(np.expand_dims(list_target, 0), np.expand_dims(pred, 0),k=top)
        result_dict[str(top)] = result
    # return NDCG_at_n(top_list, pred, target)
    return result_dict

def kendall(pred, target,mask):
    pred = torch.squeeze(pred[0][mask]).detach().cpu().numpy()
    target = target[0].data[mask].detach().cpu().numpy()
    target = target.reshape(len(target))
    #真实值排序列表
    list_target = np.zeros(len(target), dtype=int)
    for i,val in enumerate(target.argsort()):
        list_target[val] = len(target)-i
    #预测值排序列表
    list_pred = np.zeros(len(pred), dtype=int)
    for i,val in enumerate(pred.argsort()):
        list_pred[val] = len(pred)-i
    result_dict = {}
    df = pd.DataFrame({"pred":[int(m) for m in list_pred],"target":[int(j) for j in list_target]})
    result_dict["kendall"] = df.corr('kendall')['target'][0]#Kendall
    df1 = pd.DataFrame({"pred":[int(m) for m in list_pred],"target":[int(j) for j in list_target]})
    result_dict["spearman"] = df1.corr('spearman')['target'][0]#spearman
    return result_dict


if __name__ == '__main__':


    true_relevance = np.asarray([[1, 2, 3, 4, 5]])
    scores = np.asarray([[.1, .2, .3, 4, 70]])
    
    print(ndcg_score(true_relevance, scores))
    #一个参数 默认起点0，步长为1 输出：[0 1 2]
    a = np.arange(3)

    #两个参数 默认步长为1 输出[3 4 5 6 7 8]
    b = np.arange(3,9)

    #三个参数 起点为0，终点为3，步长为0.1 输出[0.  0.5 1.  1.5 2.  2.5]
    c = np.arange(0, 3, 0.5)

    df = pd.DataFrame({"pred":[1,2,3,4,5,6,7,8],"target":[2,1,3,4,6,5,8,7]})
    print(df.corr('kendall'))
    print("a----------",a)
    print("b----------",b)
    print("c----------",c)





