from experiments.experiment_setup import tpl_setup
import gnntf
import random
import numpy as np
import tensorflow as tf
import networkx as nx

with tf.device('/CPU:0'):
    gnntf.set_seed(0)
    graph, features = tpl_setup()

    #graph = nx.ego_graph(graph, list(graph)[20], 3)
    node2id = {node: i for i, node in enumerate(graph)}
    test_nodes = [node2id[node] for node in graph if "A" in node and graph.degree(node) >= 10]
    libraries = [node2id[node] for node in graph if "L" in node]
    library_set = set(libraries)
    graph = nx.relabel_nodes(graph, node2id, copy=False)
    edges = np.array([[u, v] for u, v in graph.edges()])


    print("Nodes", len(graph))
    print("Dependencies", graph.number_of_edges())
    print("Libraries", len(libraries))
    #features = np.array([[1. if node in libraries else 0.] for node in graph]) # labeling trick per https://arxiv.org/pdf/2010.16103.pdf
    features = np.zeros((len(graph), 0))
    rm = 1
    k = 5

    neighbor_edges = dict()
    for i, edge in enumerate(edges):
        if edge[0] not in neighbor_edges:
            neighbor_edges[edge[0]] = list()
        neighbor_edges[edge[0]].append(i)
        if edge[1] not in neighbor_edges:
            neighbor_edges[edge[1]] = list()
        neighbor_edges[edge[1]].append(i)
    test = [edge for node in test_nodes for edge in random.sample(neighbor_edges[node], rm)]
    non_test = list(set(range(len(edges)))-set(test))
    valid = random.sample(non_test, len(non_test)//4)
    train = list(set(range(len(edges)))-set(valid)-set(test))
    training_graph = gnntf.create_nx_graph(graph, edges[train])




    print("Training Nodes", len(training_graph))
    print("Training Dependencies", training_graph.number_of_edges())
    print("Feature shape", features.shape)

    #test_links, labels = gnntf.negative_sampling(edges[test], graph, samples=1, negative_nodes=libraries)
    #print("Test positive edges", len(edges[test]))
    #print("Test edges", len(test_links))

    link_prediction_setting = {"similarity": "dot", "loss": "diff"}
    #gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=64, latent_dims=[64], structural_dims=64)
    gnn = gnntf.GRec(gnntf.graph2adj(training_graph), features, num_classes=64, structural_dims=64)
    gnn.train(train=gnntf.LinkPrediction(lambda: gnntf.negative_sampling(edges[train], training_graph, samples=1, negative_nodes=libraries),
                                         **link_prediction_setting),
              valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph, samples=1, negative_nodes=libraries),
                                         **link_prediction_setting),
              test=gnntf.MeanLinkPrediction(edges[test], k=k, graph=graph, positive_nodes=test_nodes[:100], negative_nodes=libraries, **link_prediction_setting),
              patience=50, verbose=True, learning_rate=0.001)
