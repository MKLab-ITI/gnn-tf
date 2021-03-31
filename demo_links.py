from experiments.experiment_setup import dgl_setup, negative_sampling, graph2adj
from core.gnn import GCN, LinkPrediction
from utils import set_seed
from networkx import nx
import random
import numpy as np
import tensorflow as tf


set_seed(0)
G, labels, features, train, valid, test = dgl_setup("cora")
Gnx = nx.DiGraph([[u,v] for u,v in G.indices.numpy()])
edges = G.indices.numpy()
"""
train = random.sample(range(len(edges)), int(len(edges) * 0.8))
valid = random.sample(list(set(range(len(edges))) - set(train)), (len(edges)-len(train))//2)
test = list(set(range(len(edges))) - set(valid) - set(train))
"""
#features = tf.constant(np.eye(len(Gnx), dtype=np.float32))
#node2id = {u: idx for idx, u in enumerate(graph)}
#edges = [[node2id[u], node2id[v]] for u, v in edges]

training_graph = nx.DiGraph()
for u in Gnx:
    training_graph.add_node(u)
for u,v in edges[train]:
    training_graph.add_edge(u,v)

gnn = GCN(graph2adj(training_graph), features, num_classes=16)

gnn.train(train=LinkPrediction(*negative_sampling(edges[train], Gnx), gnn=gnn),
          valid=LinkPrediction(*negative_sampling(edges[valid], Gnx), gnn=gnn),
          test=LinkPrediction(*negative_sampling(edges[test], Gnx), gnn=gnn),
          patience=100, learning_rate=0.1)