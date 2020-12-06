import numpy as np
import networkx as nx
import random


def graph2indices(G):
    node2id = {u: idx for idx, u in enumerate(G)}
    return [[node2id[u], node2id[v]] for u, v in G.edges()]


def sample_edges(G):
    node2id = {u: idx for idx, u in enumerate(G)}
    nodes = list(G)
    edges = [[node2id[u], node2id[v]] for u, v in G.edges() if u!=v]
    labels = [1]*len(edges) + [0]*(2*len(edges))
    for u, v in G.edges():
        if u == v:
            continue
        neg = v
        while neg == u or neg == v or G.has_edge(u, neg):
            neg = random.choice(nodes)
        edges.append([node2id[u], node2id[neg]])
        neg = v
        while neg == u or neg == v or G.has_edge(neg, v):
            neg = random.choice(nodes)
        edges.append([node2id[neg], node2id[v]])
    return np.array(edges), np.array(labels)


def semisupervised_classification_setup(dataset_name, examples_per_class=20):
    G, features, labels = load(dataset_name)
    label2id = {label: idx for idx, label in enumerate(set(labels.values()))}
    labels = {u: label2id[label] for u, label in labels.items()}
    labels = np.array([labels.get(u,-1) for u in G])
    order = list(range(len(G)))
    random.shuffle(order)
    count_labels = dict()
    training_idx = list()
    for pos in order:
        if labels[pos] == -1:
            continue
        if count_labels.get(labels[pos],0) < examples_per_class:
            training_idx.append(pos)
            count_labels[labels[pos]] = count_labels.get(labels[pos], 0) + 1
    test_idx = list(set([pos for pos in range(len(G)) if labels[pos]!=-1])-set(training_idx))
    feature_size = 0
    for feat in features.values():
        feature_size = len(feat)
        break
    features = np.array([features[u] if u in features else [0 for _ in range(feature_size)] for u in G])
    return G, labels, training_idx, test_idx, features


def classification_setup(dataset_name, fraction_of_training=0.8):
    G, features, labels = load(dataset_name)
    label2id = {label: idx for idx, label in enumerate(set(labels.values()))}
    training_idx = random.sample(range(len(G)), int(len(G)*fraction_of_training))
    test_idx = list(set(range(len(G)))-set(training_idx))
    labels = {u: label2id[label] for u, label in labels.items()}
    return G, np.array([labels[u] for u in G]), training_idx, test_idx, np.array([features[u] for u in G])


def link_prediction_setup(dataset_name, fraction_of_training=0.8):
    G, features, _ = load(dataset_name)
    edges, labels = sample_edges(G)
    training_idx = random.sample(range(len(edges)), int(len(edges)*fraction_of_training))
    test_idx = list(set(range(len(edges)))-set(training_idx))
    id2nodes = dict(enumerate(G))
    for u, v in edges[test_idx]:
        u = id2nodes[u]
        v = id2nodes[v]
        if G.has_edge(u,v):
            G.remove_edge(u, v)
    return G, edges, labels, training_idx, test_idx, np.array([features[u] for u in G])


def load(dataset_name):
    G = nx.Graph()
    with open('data/'+dataset_name+'.cites') as file:
        for line in file:
            edge = line[:-1].split('\t')
            if len(edge) < 2:
                continue
            G.add_edge(edge[0], edge[1])

    features = dict()
    labels = dict()
    with open('data/'+dataset_name+'.content') as file:
        for line in file:
            line = line[:-1].split('\t')
            if line[0] not in G:
                raise Exception('Node not found '+line[0])
            features[line[0]] = [float(val) for val in line[1:-1]]
            labels[line[0]] = line[-1]

    return G, features, labels