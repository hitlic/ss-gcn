import numpy as np
from scipy.sparse import csc_matrix
import networkx as nx
import math
# import data_stackoverflow as ds
# import data_zhihu as dz
# import data_weibo as dw
import random as rand
# from bijou.metrics import NDCG_at_n,MAP,MAP_at_n
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from data import *
from metrics import *


label_key = '__label__'
feat_key = '__feat__'

def PageRank(G, s=0.85, maxerr=0.0001):
    n = G.shape[0]
    rsums = 1.0 / G.sum(axis=0)
    #print("rsums shape",rsums.shape)
    ro = np.array([0 for i in range(0,n)])
    r = np.array([1 for i in range(0,n)])
    d = np.array([s for i in range(0,n)])
    #ro, r = np.zeros(n), np.ones(n)
    
    #print("rsums shape",r.shape)
    #print("r shape",r.shape)
    # 计算PR值，直到满足收敛条件
    while np.sum(np.abs(r - ro)) > maxerr:
        ro = r.copy()
        r = np.multiply(np.multiply(d,rsums),r).dot(G)+(1-d)
        # 归一化
    #print(r.sum(1))
    return r/r.sum(1)

def norm(label_dict):
    # 标签值归一化
    l_values = [math.log(l[1] + 1) for l in label_dict.items()]  # log(x+1) 以避免x小于1导致数值为负
    l_max = max(l_values)
    l_min = min(l_values)
    for l in label_dict:
        label_dict[l] = (math.log(label_dict[l] + 1) - l_min)/(l_max - l_min)  # log(x+1) 以避免x小于1导致数值为负
    return label_dict


def regression_output(pred,label_dict,train_mask):
    # pred = normalize(pred, axis=1, norm='max')
    # pred = np.array(pred.tolist()[0])

    # target = norm(label_dict)
    target = np.squeeze(np.array([label_dict[i] for i in list(g.nodes())]))
    
    # pred = np.array([pred[i] for i in range(0,len(pred)) if train_mask[i]])
    # target = np.array([target[i] for i in range(0,len(target)) if train_mask[i]])

    # print("R2: ", '%f' % r2_score(target,pred))
    # print("MSE: ", '%f' % mean_squared_error(target,pred))
    # print("MAE: ", '%f' % np.mean(np.abs(pred - target )))
    pred = np.expand_dims(np.array(pred),axis=0)
    target = np.expand_dims(np.array(target),axis=0)
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    aa = masked_NDCG_at_n( pred, target,train_mask)
    print("=========NDCG======================")
    print(aa)
    # print("50: ", '%f' % masked_NDCG_at_n(50, pred, target,train_mask))
    # print("100: ", '%f' % masked_NDCG_at_n(100, pred, target,train_mask))
    # print("200: ", '%f' % masked_NDCG_at_n(200, pred, target,train_mask))
    # print("500: ", '%f' % masked_NDCG_at_n(500, pred, target,train_mask))
    # print("1000: ", '%f' % masked_NDCG_at_n(1000, pred, target,train_mask))
   
    # print("MAP: ", '%f' % masked_NDCG_at_n(pred,target))#g.number_of_nodes()
    print("==========MAP=====================")
    bb = masked_MAP(pred,target,train_mask)
    print(bb)
    # print("20: ", '%f' % masked_NDCG_at_n(20,pred,target,train_mask))
    # print("30: ", '%f' % masked_NDCG_at_n(30,pred,target,train_mask))
    # print("40: ", '%f' % masked_NDCG_at_n(40,pred,target,train_mask))
    # print("50: ", '%f' % masked_NDCG_at_n(50,pred,target,train_mask))
    # print("100: ", '%f' % masked_NDCG_at_n(100,pred,target,train_mask))
    # print("200: ", '%f' % masked_NDCG_at_n(200,pred,target,train_mask))
    # print("300: ", '%f' % masked_NDCG_at_n(300,pred,target,train_mask))
    # print("500: ", '%f' % masked_NDCG_at_n(500,pred,target,train_mask))
    # print("all: ", '%f' % masked_NDCG_at_n(g.number_of_nodes(),pred,target))
    print("===============================")
    cc = kendall(pred,target,train_mask)
    print(cc)


if __name__ == '__main__':

    

    g = load_graph("./datasets/arxiv_nerual_net", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    # g = load_graph("D:/exp/datasets/cora", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    # g = load_graph("D:/exp/datasets/wiki", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    # g = load_graph("D:/exp/datasets/上海地铁数据", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    #获取标签
    label = nx.get_node_attributes(g, label_key)
    # 新老id对应，新-旧
    id_dict = {}
    index = 0
    for node in g.nodes:
        id_dict[index] = node
        index = index + 1
    train_per = 0.6
    val_per = 0.2
    train_mask = [False] * g.number_of_nodes()
    val_mask = [False] * g.number_of_nodes()
    test_mask = [False] * g.number_of_nodes()
    a = b = c = 0
    for i in range(g.number_of_nodes()):
        p = rand.random()
        if p < train_per:
            if label[id_dict[i]][0] == -1.0:
                train_mask[i] = False
            else:
                train_mask[i] = True
                a=a+1
        elif train_per <= p < train_per+val_per:
            if label[id_dict[i]][0] == -1.0:
                val_mask[i] = False
            else:
                val_mask[i] = True
                b=b+1
        else:
            if label[id_dict[i]][0] == -1.0:
                test_mask[i] = False
            else:
                test_mask[i] = True
                c=c+1
    print(a,"!!!   ",b,"!!   ",c)
    # mask = [False] * g.number_of_nodes()
    
    
    # for node in g.nodes:
    #     if label[node][0] != -1:
    #         mask[id_dict[node]] = True
    # GA = nx.to_numpy_matrix(g,nodelist=sorted(g.nodes()))
    pred1 = nx.pagerank(g,alpha=0.75)
    pred_result = []
    for key,val in pred1.items():
        pred_result.append(val)
    
    # pred = PageRank(GA, s=0.85, maxerr=0.0001)
    regression_output(np.array(pred_result),label,test_mask)
    
    
