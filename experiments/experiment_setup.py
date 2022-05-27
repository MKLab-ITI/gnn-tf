import numpy as np
import networkx as nx
import random
import pickle


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


def cite_setup(name, seed=0):
    G, features, labels = load(name)
    print(len(features))
    features = np.array(features)
    labels = np.array(labels)
    train, valid, test = custom_splits(labels, num_validation=500, seed=seed)
    return G, labels, features, train, valid, test


def split_to_words(sentence):
    if "_" in sentence:
        ret = list()
        for word in sentence.split("_"):
            ret += split_to_words(word)
        return ret
    if "." in sentence:
        ret = list()
        for word in sentence.split("."):
            ret += split_to_words(word)
        return ret
    for pos, letter in enumerate(sentence):
        if pos > 0 and letter.isupper() and sentence[pos-1].islower() and (pos<2 or sentence[pos-2].islower()):
            return split_to_words(sentence[:pos])+split_to_words(sentence[pos:])
    return [sentence.lower()]


def tpl_setup(path="data"):
    G = nx.DiGraph()
    with open(path+'/relation.txt') as file:
        for line in file:
            edge = line[:-1].split(',')
            if len(edge) < 2:
                continue
            u = "A"+edge[-2].split(":")[-1]
            v = "L"+edge[-1].split(":")[-1]
            G.add_edge(u, v)

    nodes = set([v for v in G if "A" in v and G.out_degree(v)>=10])
    Gprev = G
    G = nx.DiGraph()
    for u, v in Gprev.edges():
        if u in nodes:
            G.add_edge(u, v)

    features = dict()
    with open(path+'/apk_info.csv') as file:
        for line in file:
            line = line[:-1].split(',')
            line[0] = "A"+line[0]
            if line[0] not in G:
                continue
            features[line[0]] = split_to_words(line[1])
    with open('data/lib_info.csv') as file:
        for line in file:
            line = line[:-1].split(',')
            line[0] = "L"+line[0]
            if line[0] not in G:
                continue
            features[line[0]] = split_to_words(line[1])
    occurences = dict()
    for words in features.values():
        for word in words:
            occurences[word] = occurences.get(word, 0) + 1
    feature2id = dict()
    for words in features.values():
        for word in words:
            if word not in feature2id and occurences[word] > 3:
                feature2id[word] = len(feature2id)
    #entries = list()
    feature_matrix = np.zeros((len(G), len(feature2id)), dtype=np.float32)
    for row, node in enumerate(G):
        for word in features[node]:
            if word in feature2id:
                feature_matrix[row,feature2id[word]] = 1.
                #entries.append([row,feature2id[word], 1.])
    #print("number of words", len(feature2id))
    #feature_matrix = scipy.sparse.csr_matrix(entries, shape=(len(G), len(feature2id)))

    return G, np.array(feature_matrix)


def maven_setup():
    """
    path = "data/"
    pairs = list()
    features = dict()

    with open(path + 'links_all.csv', 'r') as file:
        next(file)  # skips first line
        i = 0
        features = dict()
        for line in file:
            i += 1
            entries = line[:-1].split(",")
            entries[0] = ":".join(entries[0].split(":")[:-1])
            entries[1] = ":".join(entries[1].split(":")[:-1])

            pairs.append((anonymizer[entries[0]], anonymizer[entries[1]]))

    G = nx.DiGraph()
    for u in features:
        G.add_nodes(u)
    for u, v in pairs:
        G.add_edge(u, v)
    return G, None
    """
    return None


def dgl_setup(dataset_name):
    print(dataset_name+".dat")
    import os.path
    if os.path.exists("data/"+dataset_name+".dat"):
        return pickle.load(open("data/"+dataset_name+".dat", "rb"))
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
    ret = (G, labels.numpy(), features.numpy(), np.where(train_mask)[0].tolist(), np.where(val_mask)[0].tolist(), np.where(test_mask)[0].tolist())
    pickle.dump(ret, open("data/"+dataset_name+".dat", "wb"))
    return ret

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