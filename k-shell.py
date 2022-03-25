import networkx as nx
import matplotlib.pyplot as plt
import math

from networkx.algorithms.shortest_paths.unweighted import predecessor
from data import *
import torch
from metrics import *
from sklearn import preprocessing 
import random as rand
label_key = '__label__'

#抽取txt中的数据
def read_txt(data):
    g = nx.read_weighted_edgelist(data)
    print(g.edges())
    return g
def gDegree(G):
    """
    将G.degree()的返回值变为字典
    """
    node_degrees_dict = {}
    for i in G.degree():
        node_degrees_dict[i[0]]=i[1]
    return node_degrees_dict.copy()


def kshell(G):
    """
    kshell(G)计算k-shell值
    """
    graph = G.copy()
    importance_dict = {}
    ks = 1
    while graph.nodes():
        temp = []
        node_degrees_dict = gDegree(graph)
        kks = min(node_degrees_dict.values())
        while True:
            for k, v in node_degrees_dict.items(): 
                if v == kks:
                    temp.append(k)
                    graph.remove_node(k)
                    node_degrees_dict = gDegree(graph)
            if kks not in node_degrees_dict.values():
                break
        importance_dict[ks] = temp
        ks += 1
    return importance_dict
def sumD(G):
    """
    计算G中度的和
    """
    G_degrees = gDegree(G)
    sum = 0
    for v in G_degrees.values():
        sum += v
    return sum
def getNodeImportIndex(G):
    """
    计算节点的重要性指数
    """
    sum = sumD(G)
    I = {}
    G_degrees = gDegree(G)
    for k,v in G_degrees.items():
        I[k] = v/sum
    return I

def Entropy(G):
    """
    Entropy(G) 计算出G中所有节点的熵
    I 为重要性
    e 为节点的熵sum += I[i]*math.log(I[i])
    """
    I = getNodeImportIndex(G)
    e = {}
    for k,v in I.items():
        sum = 0
        for i in G.neighbors(k):
            sum += I[i]*math.log(I[i])
        sum = -sum
        e[k] = sum
    return e

def kshellEntropy(G):
    """
    kshellEntropy(G) 是计算所有壳层下，所有节点的熵值
    例：
    {28: {'1430': 0.3787255719932099,
          '646': 0.3754626894107377,
          '1431': 0.3787255719932099,
          '1432': 0.3787255719932099,
          '1433': 0.3754626894107377
          ....
    ks is a dict 显示每个壳中的节点
    e 计算了算有节点的熵 
    """
    ks = kshell(G)
    e = Entropy(G)
    ksES = {}
    ksI = max(ks.keys())
    while ksI > 0:
        ksE = {}
        for i in ks[ksI]:
            ksE[i] = e[i]
        ksES[ksI] = ksE
        ksI -= 1
    return ksES

def kshellEntropySort(G):
    ksE = kshellEntropy(G)
    ksES = []
    ksI = max(ksE.keys())
    while ksI > 0:
        t = sorted([(v, k) for k, v in ksE[ksI].items()],reverse=True)
        ksES.append(list(i[1] for i in t))
        ksI -= 1
    return ksES

def getRank(G):
    rank = []
    rankEntropy = kshellEntropySort(G)
    while (len(rankEntropy)!= 0):
        for i in range(len(rankEntropy)):
            rank.append(rankEntropy[i].pop(0))
        while True:
            if [] in rankEntropy:
                rankEntropy.remove([])
            else:
                break
    return rank

def normalize(label_dict):
    # 标签值归一化
    l_values = [math.log(l[1] + 1) for l in label_dict.items()]  # log(x+1) 以避免x小于1导致数值为负
    l_max = max(l_values)
    l_min = min(l_values)
    for l in label_dict:
        label_dict[l] = (math.log(label_dict[l] + 1) - l_min)/(l_max - l_min)  # log(x+1) 以避免x小于1导致数值为负
    return label_dict

 
if __name__ == '__main__':
    
    # g = load_graph("./datasets/cora", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    g = load_graph("./datasets/co-auth", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    # g = load_graph("./datasets/wiki", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    # g = load_graph("./datasets/metro", g_delimiter = ',', feat_delimiters = (':', ','), head = True)
    #排序的结果
    
    #rank = getRank(g)
    ks_dict = kshell(g)
    pred_dict = {}
    for key,val in ks_dict.items():
        for node in val:
            pred_dict[node] = key
    
    aa = np.array(list(pred_dict.values()),dtype=float).reshape(-1, 1)
    # pred_first = kshell(g)
    min_max_scaler = preprocessing.MinMaxScaler() 
  
    X_minMax = min_max_scaler.fit_transform(aa)
    label_dict = nx.get_node_attributes(g, label_key)
    #得到一一对应的预测值和真实值，记住编号
    pred = []
    target = []
    id_dict = {}
    index = 0
    for key,value in pred_dict.items():
        pred.append(value)
        target.append(label_dict[key])
        id_dict[index] = key
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
            if label_dict[id_dict[i]][0] == -1.0:
                train_mask[i] = False
            else:
                train_mask[i] = True
                a=a+1
        elif train_per <= p < train_per+val_per:
            if label_dict[id_dict[i]][0] == -1.0:
                val_mask[i] = False
            else:
                val_mask[i] = True
                b=b+1
        else:
            if label_dict[id_dict[i]][0] == -1.0:
                test_mask[i] = False
            else:
                test_mask[i] = True
                c=c+1
    print(a,"!!!   ",b,"!!   ",c)


    # mask = [False] * g.number_of_nodes()
    # for node in g.nodes:
    #     if label_dict[node][0] != -1:
    #         mask[id_dict[node]] = True

    pred = np.expand_dims(np.array(pred),axis=0)
    target = np.expand_dims(np.array(target),axis=0)
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    
    print("============nDCG===================")
    nDCG = masked_NDCG_at_n(pred, target,test_mask)
    print(nDCG)
    print("==========map=====================")
    map = masked_MAP(pred,target,test_mask)
    print(map)
    print("==========kendall=====================")
    cc = kendall(pred,target,train_mask)
    print(cc)
