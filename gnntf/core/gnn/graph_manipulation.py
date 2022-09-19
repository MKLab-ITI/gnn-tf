import tensorflow as tf
import networkx as nx


def create_nx_graph(nodes, edges):
    graph = nx.DiGraph()
    if nodes is not None:
        for u in nodes:
            graph.add_node(u)
    for u, v in edges:
        graph.add_edge(u, v)
    return graph


def adj2graph(nodes, adj):
    return create_nx_graph(nodes, adj.indices.numpy())


def graph2indices(G):
    node2id = {u: idx for idx, u in enumerate(G)}
    return [[node2id[u], node2id[v]] for u, v in G.edges()]


def graph2adj(G, directed=False):
    #indices = [[u, v] for u, v in G.edges()]#
    indices = graph2indices(G)
    values = [edge[2].get("weight", 1.) for edge in G.edges(data=True)]  # [1. for _ in range(len(indices))]
    if not directed:
        indices = indices + [[v, u] for u, v in indices]
        values = values + values
    return tf.sparse.SparseTensor(indices, values, (len(G), len(G)))