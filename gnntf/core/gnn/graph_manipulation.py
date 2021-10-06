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
    if not directed:
        for u,v in list(G.edges()):
            if not G.has_edge(v,u):
                G.add_edge(v,u)
    values = [1. for _ in G.edges()]
    return tf.sparse.SparseTensor(graph2indices(G), values, (len(G), len(G)))