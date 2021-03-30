import numpy as np
import networkx as nx
import random
import tensorflow as tf


def enrich_features(features, positional=True, labels=None, train=None):
    if labels is not None:
        num_classes = len(set(labels.toarray()))
        label_features = np.zeros((features.shape[0], num_classes))
        for i in train:
            label_features[i][labels[i]] = 1
        features = np.concatenate((features, label_features), axis=1)
    if positional:
        num_positions = int(1.5 + np.log2(features.shape[0] + 1))
        positional_fatures = np.zeros((features.shape[0], num_positions))
        for i in range(features.shape[0]):
            norm = sum(float(val) for val in bin(i + 1)[2:])
            for pos, val in enumerate(bin(i + 1)[2:]):
                positional_fatures[i][positional_fatures.shape[1] - 1 - pos] = float(val) / norm
        features = np.concatenate((features, positional_fatures), axis=1)
    return features


def graph2indices(G):
    node2id = {u: idx for idx, u in enumerate(G)}
    return [[node2id[u], node2id[v]] for u, v in G.edges()]


def negative_sampling(positive_edges, graph, samples=10):
    edges = list()
    values = list()
    nodes = list(graph)
    for u, v in positive_edges:
        edges.append([u, v])
        values.append(1)
        for _ in range(samples):
            vneg = u
            while graph.has_edge(u, vneg) or graph.has_edge(vneg, u) or vneg == u:
                vneg = random.choice(nodes)
            edges.append([u, vneg])
            values.append(0)

    return np.array(edges), values


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


def graph2adj(G):
    for u,v in list(G.edges()):
        if not G.has_edge(v,u):
            G.add_edge(v,u)
    values = [1. for _ in G.edges()]
    return tf.sparse.SparseTensor(graph2indices(G), values, (len(G), len(G)))

def cite_setup(name, seed=0):
    G, features, labels = load(name)
    print(len(features))
    features = np.array(features)
    labels = np.array(labels)
    train, valid, test = custom_splits(labels, num_validation=500, seed=seed)
    return graph2adj(G), labels, features, train, valid, test

def dgl_setup(dataset_name):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    if dataset_name == "cora":
        data = CoraGraphDataset(verbose=False)
    elif dataset_name == "citeseer":
        data = CiteseerGraphDataset(verbose=False)
    elif dataset_name == "pubmed":
        data = PubmedGraphDataset(verbose=False)
    else:
        raise Exception("Invalid dataset name")
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    G = nx.DiGraph()
    U, V = g.edges()
    for u in g.nodes().numpy().tolist():
        G.add_node(u)
    for u, v in zip(U.numpy().tolist(), V.numpy().tolist()):
        G.add_edge(u, v)
    return graph2adj(G), labels.numpy(), features.numpy(), np.where(train_mask)[0].tolist(), np.where(val_mask)[0].tolist(), np.where(test_mask)[0].tolist()

def custom_splits(labels, examples_per_class=20, num_validation=500, seed=0):
    random.seed(seed)
    order = list(range(labels.shape[0]))
    random.shuffle(order)
    count_labels = dict()
    training_idx = list()
    for pos in order:
        if labels[pos] == -1:
            continue
        if count_labels.get(labels[pos], 0) < examples_per_class:
            training_idx.append(pos)
            count_labels[labels[pos]] = count_labels.get(labels[pos], 0) + 1
    test_idx = list(set([pos for pos in range(labels.shape[0]) if labels[pos] != -1]) - set(training_idx))
    random.shuffle(test_idx)
    if num_validation is None:
        num_validation = len(count_labels)*examples_per_class
    valid_idx = test_idx[:num_validation]
    test_idx = test_idx[num_validation:]
    return training_idx, valid_idx, test_idx

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


_loaded = dict()
def load(dataset_name):
    if dataset_name in _loaded:
        G, features, labels = _loaded[dataset_name]
        return G.copy(), features, labels
    if '.npz' in dataset_name:
        G, features, labels = __np_load(dataset_name)
    else:
        G, features, labels = __dataload(dataset_name)
    _loaded[dataset_name] = (G, features, labels)
    return G, features, labels


def __np_load(dataset_name):
    from scipy.sparse import csr_matrix
    loc = np.load('data/'+dataset_name,allow_pickle=True)
    adj_matrix = csr_matrix((loc['adj_matrix.data'], loc['adj_matrix.indices'], loc['adj_matrix.indptr']), shape=loc['adj_matrix.shape'], dtype=float)
    attr_matrix = csr_matrix((loc['attr_matrix.data'], loc['attr_matrix.indices'], loc['attr_matrix.indptr']), shape=loc['attr_matrix.shape'], dtype=float)
    G = nx.convert_matrix.from_scipy_sparse_matrix(adj_matrix, create_using=nx.DiGraph)
    attr_matrix = attr_matrix.todense().tolist()
    features = {u: attr_matrix[u] for u in range(len(G))}
    labels = {u: label for u, label in enumerate(loc['labels'])}
    return G, features, labels


def __dataload(dataset_name):
    G = nx.DiGraph()
    with open('data/'+dataset_name+'.cites') as file:
        for line in file:
            edge = line[:-1].split('\t')
            if len(edge) < 2:
                continue
            u = edge[-2].split(":")[-1]
            v = edge[-1].split(":")[-1]
            if u != v:
                G.add_edge(u, v)
    features = dict()
    labels = dict()
    feature_map = None
    with open('data/'+dataset_name+'.content') as file:
        for line in file:
            line = line[:-1].split('\t')
            if "NODE"==line[0]:
                continue
            if ":label" in line[0]:
                feature_map = [var.split(":")[1] for var in line[2:]]
                continue
            if line[0] not in G:
                continue
                #raise Exception('Node not found '+line[0])
            if feature_map is not None:
                line_feats = {val.split("=")[0]: val.split("=")[1] for val in line[2:]}
                line_feats["summary"] = 0
                features[line[0]] = [float(line_feats.get(val, 0)) for val in feature_map]
                labels[line[0]] = line[1]
            else:
                features[line[0]] = [float(val) for val in line[1:-1]]
                labels[line[0]] = line[-1]
    for u in list(G):
        if u not in features:
            G.remove_node(u)
    sums = {u: 1 for u in G}#sum(features[u]) for u in G}
    features = {u: [f/sums[u] if f!=0 else 0 for f in features[u]] for u in G}
    return G, features, labels