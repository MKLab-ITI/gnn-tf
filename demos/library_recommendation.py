from experiments.experiment_setup import tpl_setup
import gnntf
import random
import numpy as np
import tensorflow as tf
import networkx as nx

with tf.device('/CPU:0'):
    gnntf.set_seed(0)
    graph, features = tpl_setup()

    node_order = [node for node in graph if "A" in node]+[node for node in graph if "L" in node]
    node2id = {node: i for i, node in enumerate(node_order)}
    users = [node2id[node] for node in graph if "A" in node]
    items = [node2id[node] for node in graph if "L" in node]
    edges = [[node2id[u], node2id[v]] for u, v in graph.edges()]
    graph = gnntf.create_nx_graph(node_order, edges)
    item_set = set(items)
    edges = np.array(edges)


    print("Nodes", len(users))
    print("Users", len(edges))
    print("Items", len(items))

    #features = np.array([[1. if node in libraries else 0.] for node in graph]) # labeling trick per https://arxiv.org/pdf/2010.16103.pdf
    features = np.zeros((len(graph), 0))
    rm = 1
    k = 5

    user2items = dict()
    for i, edge in enumerate(edges):
        if edge[0] not in user2items:
            user2items[edge[0]] = list()
        user2items[edge[0]].append(i)
        if edge[1] not in user2items:
            user2items[edge[1]] = list()
        user2items[edge[1]].append(i)

    test = [edge for user in users for edge in random.sample(user2items[user], rm)]
    non_test = list(set(range(len(edges)))-set(test))
    valid = random.sample(non_test, len(non_test)//4)
    train = list(set(range(len(edges)))-set(valid)-set(test))
    training_graph = gnntf.create_nx_graph(graph, edges[non_test])

    print("Training links", training_graph.number_of_edges())
    print("Feature shape", features.shape)
    preprocessor = gnntf.Structural(dims=16, regularize=0.2, l2_contraint=False, bipartite=len(users))
    #gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=64, latent_dims=[64], preprocessor=preprocessor)
    gnn = gnntf.NGCF(gnntf.graph2adj(training_graph), features, num_classes=64, preprocessor=preprocessor)

    batch_size = 2048
    gnn.train(train=gnntf.LinkPrediction(lambda: gnntf.negative_sampling(edges[random.sample(train, batch_size)], training_graph, samples=1, negative_nodes=items)),
              valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph, samples=1, negative_nodes=items)),
              test=gnntf.MeanLinkPrediction(edges[test], k=k, graph=graph, positive_nodes=users[:100], negative_nodes=items),
              patience=50, verbose=True, learning_rate=0.01, batches=len(train)//batch_size+1)
