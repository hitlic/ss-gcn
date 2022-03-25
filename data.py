import networkx as nx
import warnings
from pathlib import Path
import torch
from torch_geometric.data import Data
import random as rand
from split_graph_by_metis import *
import math

feat_key = '__feat__'
label_key = '__label__'


def load_edges(path='edges', delimiter=',', head=False):
    """
    从边文件中加载网络的边。文件格式如下（#表示注释符号）：
        # 表示注释
        a, b  # 边(a,b)
        a, c  # 边(a,c)
        ...
    Args:
        path: 存储边的文件名，默认为'edges'
        delimiter: 分割符，默认为','
        head: 是否包含标题行
    Return: networkx.Graph
    """
    print('Loading edge data ...')
    edges = []
    skip_head = 0
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line.strip():
                continue
            #这里是一个列表
            endpoints = line.split(delimiter)
            assert len(endpoints) == 2, 'Invalid Format! Each edge must contains two nodes!'
            if head and skip_head == 0:
                skip_head = skip_head + 1
                continue
            endpoints = [i for i in endpoints]
            edges.append(tuple(endpoints))
    # if head:
    #     edges = edges[1:]
    return edges


def load_features(path='features', delimiters=(':', ','), head=False):
    """
    文件格式：
        # 表示注释
        node1: f11, f12, f13, ...   # 节点node1的特征
        node2: f21, f22, f23, ...
        ...
    Args:
        path: 文件名
        delimiters: 节点-特征分割符，特征分割符
    Return: 特征字典，key为节点id，value为特征
    """
    print('Loading feature data ...')
    features = dict()
    skip_head = 0
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line.strip():
                continue

            if head and skip_head == 0:  # skip head line
                skip_head += 1
                continue

            node_feat = line.split(delimiters[0], 1)
            assert len(node_feat) == 2, 'Invalid Format! Each line must contains two patrs of node and feature.'
            features[node_feat[0]] = [float(f) for f in node_feat[1].split(delimiters[1])]
            data = np.array(features[node_feat[0]])
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            if std != 0:
                data = (data - mean)/std
            features[node_feat[0]] = data.tolist()

    feature_lens = {len(feat) for _, feat in features.items()}
    if len(feature_lens) > 1:
        warnings.warn('Features have different lengthes!')
    
    return features

def load_label(path='labels', delimiters=(':', ','), head=False):
    """
    文件格式：
        # 表示注释
        node1: f1   # 节点node1的标签
        node2: f2
        ...
    Args:
        path: 文件名
        delimiters: 节点-标签分割符，标签分割符
    Return: 标签字典，key为节点id，value为标签
    """
    print('Loading labels data ...')
    labels = dict()
    skip_head = 0
    with open(path) as f:
        #循环文件的每一行
        for line in f:
            #strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            line = line.split('#', 1)[0].strip()
            if not line.strip():
                continue

            if head and skip_head == 0:  # skip head line
                skip_head += 1
                continue

            node_label = line.split(delimiters[0], 1)
            assert len(node_label) == 2, 'Invalid Format! Each line must contains two patrs of node and feature.'
            for f in node_label[1].split(delimiters[1]):
                if f != '0':
                    labels[node_label[0]]  = [math.log(float(f))]
            # labels[node_label[0]] = [float(f) for f in node_label[1].split(delimiters[1])]
    label_lens = {len(label) for _, label in labels.items()}
    if len(label_lens) > 1:
        warnings.warn('labels have different lengthes!')
    return labels


def load_graph(path, g_delimiter=',', feat_delimiters=(':', ','), head=False):
    """
    从文件中加载图数据。
    Args:
        path: 两种取值，a)文件夹地址，此时默认的边文件名为edges，特征文件名为features；b)取值为元组(边文件地址, 特征文件地址)
        g_delimiter: 边文件分割符
        feat_delimiters: 特征文件分割符
        head: 首行是否标题
    Return:
        networkx.Graph with node features
    """
    if type(path) in [list, tuple]:
        assert len(path) == 2, '需提供(边索引文件, 特征文件)'
        path_edges, path_feats = path
    else:
        path = Path(path)
        path_edges, path_feats,path_labels = path/'edges', path/'features',path/'labels'

    edges = load_edges(path_edges, g_delimiter, head)
    feats_dict = load_features(path_feats, feat_delimiters, head) if path_feats else None
    label_dict = load_label(path_labels, feat_delimiters, head) if path_labels else None
    g = nx.Graph()  
    g.add_edges_from(edges)
    
    # 合并网络和节点特征
    for n in g.nodes:
        g.nodes[n][feat_key] = feats_dict[n]
        if n in label_dict.keys():
            g.nodes[n][label_key] = label_dict[n]
        else:
            g.nodes[n][label_key] = [float(-1)]
    return g

#得到特征
def load_feat(g):
    #获取到所有的特征了
    #获取网络的度,然后拼到节点特征后面
    d = nx.degree(g)
    core_num = nx.core_number(g)
        #图的局部聚类系数
    cluster_coeff = nx.clustering(g)
        #计算图的Collective Influence（CI）
    ci = {}
    for n in g.nodes:
        one_hop_nodes_iter_dic = g[n]
        degree_sum = 0
        if g.degree(n)-1 == 0:
            ci[n] = 0
        else:
            for node in one_hop_nodes_iter_dic.__iter__():
                degree_sum = degree_sum + (g.degree(node) - 1)
            ci[n] = (g.degree(n)-1) * degree_sum
    return d,core_num,cluster_coeff,ci

#把网络数据包装成pyg的data
def load_pygdata(path, g_delimiter=',', feat_delimiters=(':', ','), head=False,train_per=0.6, val_per=0.2):
    g = load_graph(path, head = True)
    #给结点重新编号，新旧id对应存储字典
    node2index = {n:i for i,n in enumerate(g.nodes)}
    #获取网络中边连接的节点，返回的是两个元组（元组中的元素不可更改）,两个元组中的节点一一对应，即一条边
    s_nodes, t_nodes = zip(*g.edges)
    #给新编号和旧编号对应上，利用新编号来构造边的列表
    s_re_nodes = [node2index[n] for n in s_nodes]
    t_re_nodes = [node2index[n] for n in t_nodes]
    source = list(s_re_nodes) + list(t_re_nodes)
    target = list(t_re_nodes) + list(s_re_nodes)
    #构造pyg中data数据中的边，双向，这里可以理解为无向图相当于双向图，因此两边连接
    edges = torch.tensor([source, target], dtype=torch.long)

    #构建新图--预训练任务用
    new_edges = [[edge,t_re_nodes[i]] for i,edge in enumerate(s_re_nodes)]
    graph = nx.Graph()
    graph.add_edges_from(new_edges)
    #预训练标签
    adj_list = [list(graph.adj[node]) for node in graph.nodes]
    # adj_result = [[node2index[j] for j in val] for val in adj_list]
    ssl_agent = ClusteringMachine(graph,adj_list)
    ssl_agent.decompose()
    dis_matrix = ssl_agent.dis_matrix

    #构造节点属性及标签
    node_list = list(g.nodes)
    feat_dict = nx.get_node_attributes(g, feat_key)
    x = [feat_dict[n] for n in node_list]
    x = torch.tensor(x, dtype=torch.float)
    #有标签的节点也要跟节点的新编号对上,按照新id重编号
    label_dict = nx.get_node_attributes(g, label_key)
    
    y = [label_dict[n] for n in node_list]
    y = torch.tensor(y, dtype=torch.float)#floatlong
    
    data = Data(x = x, edge_index = edges, y = y, y_dis = dis_matrix)
    #创建一个列表，列表里全是false，长度为节点个数，但值为-1的都置为False
    train_mask = [False] * g.number_of_nodes()
    val_mask = [False] * g.number_of_nodes()
    test_mask = [False] * g.number_of_nodes()
    a = b = c = 0
    for i in range(g.number_of_nodes()):
        p = rand.random()
        if p < train_per:
            if data.y[i] == -1.0:
                train_mask[i] = False
            else:
                train_mask[i] = True
                a=a+1
        elif train_per <= p < train_per+val_per:
            if data.y[i] == -1.0:
                val_mask[i] = False
            else:
                val_mask[i] = True
                b=b+1
        else:
            if data.y[i] == -1.0:
                test_mask[i] = False
            else:
                test_mask[i] = True
                c=c+1
    print(a,"!!!   ",b,"!!   ",c)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    return data



if __name__ == '__main__':
    es = [(0, 2), (0, 3), (0, 4),
          (1, 4), (1, 5), (1, 6),
          (2, 3), (4, 5), (2, 7), (2, 8),
          (3, 8), (3, 9), (3, 10),
          (4, 10), (4, 11),
          (5, 11), (5, 12), (5, 13),
          (6, 13), (6, 14),
          (7, 8), (9, 10), (11, 12), (13, 14),
          (7, 15), (7, 16), (8, 16), (8, 17), (9, 17), (9, 18), (10, 18), (10, 19),
          (11, 19), (11, 20), (12, 20), (12, 21), (13, 21), (13, 22), (14, 22)
          ]
    # es = load_edges('cites.csv', head=True)
    # # es = load_edges('edges', head=True)
    # # feat = load_features(head=True)
    # g = nx.Graph()
    # g.add_edges_from(es)

    #load_label(path='./datasets/Cora/labels', delimiters=(':', ','), head=True)

    #创建一个图
    g = nx.Graph()
    g.add_node('a',name = "test")
    g.add_nodes_from(['b','c','d','e','f'])
    g.add_edges_from([('a', 'b'), ('a', 'c'),('a', 'e'),('b', 'd'),('b', 'a'),('c', 'a'),('d', 'b'),('e', 'a'),('f','b')])
    d,core_num,cluster_coeff,ci = load_feat(g)
    #g = load_graph('./datasets/Cora', head=True)
