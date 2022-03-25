import pymetis
import networkx as nx
import collections
import numpy as np
import torch
from sklearn.model_selection import train_test_split


# nodes=[0,1,2,3,4,5]
# edges=[(0,1),(0,3),(0,5),(1,2),(1,4),(2,4)]
# G = nx.Graph()

#往图添加节点和边
# G.add_nodes_from(nodes)
# G.add_edges_from(edges)
# adj = nx.adjacency_matrix(G)
# graph = nx.from_scipy_sparse_matrix(adj)
# #判断两个图是否一致(不考虑边属性)，事实证明，根据邻接矩阵生成的图和之前的图是一样的
# nx.is_isomorphic(G, graph)
# # adj = [list(G.adj[node]) for node in G.nodes]
# (st, parts) = pymetis.part_graph(3,G)


class ClusteringMachine(object):
#    def __init__(self, adj, features, cluster_number=20, clustering_method='metis'):
    def __init__(self,graph, adj_result,cluster_number = 3, clustering_method='metis'):
        self.adj = nx.adjacency_matrix(graph)
        #   self.features = features.cpu().numpy()
        #从稀疏矩阵（也就是邻接矩阵）得到图 --- 跟直接把图传过来有什么区别
        # self.graph = nx.from_scipy_sparse_matrix(adj)
        self.clustering_method = clustering_method
        self.cluster_number = cluster_number
        self.graph = graph
        self.adj_result = adj_result

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = pymetis.part_graph(self.cluster_number,self.adj_result)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def get_central_nodes(self):
        """
        set the central node as the node with highest degree in the cluster
        """
        self.general_data_partitioning()
        central_nodes = {}
        for cluster in self.clusters:
            counter = {}
            for node, _ in self.sg_edges[cluster]:
                counter[node] = counter.get(node, 0) + 1
            sorted_counter = sorted(counter.items(), key=lambda x:x[1])
            central_nodes[cluster] = sorted_counter[-1][0]
        return central_nodes

    def transform_depth(self, depth):
        return 1 / depth

    def shortest_path_to_clusters(self, central_nodes, transform=False):
        """
        Do BFS on each central node, then we can get a node set for each cluster
        which is within k-hop neighborhood of the cluster.
        """
        # self.distance = {c:{} for c in self.clusters}
        self.dis_matrix = -np.ones((self.adj.shape[0], self.cluster_number))
        for cluster in self.clusters:
            node_cur = central_nodes[cluster]
            visited = set([node_cur])
            q = collections.deque([(x, 1) for x in self.graph.neighbors(node_cur)])
            while q:
                node_cur, depth = q.popleft()
                if node_cur in visited:
                    continue
                visited.add(node_cur)
                # if depth not in self.distance[cluster]:
                #     self.distance[cluster][depth] = []
                # self.distance[cluster][depth].append(node_cur)
                if transform:
                    self.dis_matrix[node_cur][cluster] = self.transform_depth(depth)
                else:
                    self.dis_matrix[node_cur][cluster] = depth
                for node_next in self.graph.neighbors(node_cur):
                    q.append((node_next, depth+1))

        if transform:
            self.dis_matrix[self.dis_matrix==-1] = 0
        else:
            self.dis_matrix[self.dis_matrix==-1] = self.dis_matrix.max() + 2
        return self.dis_matrix

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = 0.8)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
        print('Number of nodes in clusters:', {x: len(y) for x,y in self.sg_nodes.items()})

    def decompose(self):
        """
        Decomposing the graph, partitioning, creating Torch arrays.
        """
        if self.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
            central_nodes = self.get_central_nodes()
            self.shortest_path_to_clusters(central_nodes)
        elif self.clustering_method == "kmedoids":
            print("\nKMedoids node clustering started.\n")
            central_nodes = self.kmedoids_clustering()
            self.shortest_path_to_clusters(central_nodes)
        elif self.clustering == "random":
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
            central_nodes = self.get_central_nodes()
            self.shortest_path_to_clusters(central_nodes)

        self.dis_matrix = torch.FloatTensor(self.dis_matrix)
