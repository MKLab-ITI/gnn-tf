import tensorflow as tf
import numpy as np
from gnntf.core.nn.layers import Layered
from gnntf.core.nn import Predictor
import random
from gnntf import measures
from tqdm import tqdm


class NodeClassification(Predictor):
    def __init__(self, nodes, labels=None, loss_transform=None):
        self.nodes = nodes
        self.labels = labels
        self.loss_transform = loss_transform

    def predict(self, features: tf.Tensor):
        return tf.argmax(tf.nn.embedding_lookup(features, self.nodes), axis=1) # data is a list of nodes

    def loss(self, features: tf.Tensor):
        if self.labels is None:
            raise Exception("Evaluation requires node labels")
        if self.loss_transform is not None:
            features = self.loss_transform(features)
        predictions = tf.nn.log_softmax(tf.nn.embedding_lookup(features, self.nodes), axis=1)
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(self.labels, predictions)

    def evaluate(self, features: tf.Tensor):
        if self.labels is None:
            raise Exception("Evaluation requires node labels")
        predictions = tf.argmax(tf.nn.embedding_lookup(features, self.nodes), axis=1)
        return 1-tf.math.count_nonzero(predictions-self.labels).numpy()/predictions.shape[0]


def recommend_all(node, graph=None, positive_edges=None, negative_nodes=None):
    edges = list()
    values = list()
    if positive_edges is None:
        positive_edges = list([node, neighbor] for neighbor in graph.neighbors(node))
    if negative_nodes is None:
        negative_nodes = list(graph)
    for u, v in positive_edges:
        edges.append([u,v])
        values.append(1)
    for v in negative_nodes:
        if v != node and (graph is None or (not graph.has_edge(node, v) and not graph.has_edge(v, node))):
            edges.append([node,v])
            values.append(0)
    return np.array(edges), values


def negative_sampling(positive_edges, graph, samples=1, negative_nodes=None):
    edges = np.empty((2*len(positive_edges),2), dtype=int)
    if negative_nodes is None:
        negative_nodes = list(graph)
    i = 0
    for u, v in positive_edges:
        edges[i, 0], edges[i, 1] = u, v
        i += 1
        for _ in range(samples):
            vneg = v
            while vneg == u or vneg == v or graph.has_edge(u, vneg) or graph.has_edge(vneg, u):
                vneg = random.choice(negative_nodes)
            edges[i, 0], edges[i, 1] = u, vneg
            i += 1
    return edges, np.tile(np.array([1.]+[0.]*samples), len(positive_edges))


class LinkPrediction(Predictor):
    def __init__(self, edges, labels=None, gnn: Layered=None, similarity="dot", loss="diff", regularize=0):
        if callable(edges):
            self.edge_sampler = edges
            edges, labels = edges()
        else:
            self.edge_sampler = None
        self.edges = edges
        self.loss_func = loss
        self.labels = None if labels is None else tf.constant(labels, shape=(len(labels),1))
        self.r = None if gnn is None else gnn.create_var(shape=(gnn.top_shape()[1],1), regularize=0, shared_name="distmult", normalization="ones", trainable=True)
        self.similarity = similarity
        self.regularize = regularize

    def _update_labels(self):
        if self.edge_sampler is not None:
            edges, labels = self.edge_sampler()
            self.edges = edges
            self.labels = None if labels is None else tf.constant(labels, shape=(len(labels),1))

    def predict(self, features: tf.Tensor, to_logits=False):
        self._update_labels()
        if self.similarity == "cos":
            features = tf.math.l2_normalize(features, axis=1)
        similarities = tf.multiply(tf.nn.embedding_lookup(features, self.edges[:,0]), tf.nn.embedding_lookup(features, self.edges[:,1]))
        logits = tf.reduce_sum(similarities, axis=1) if self.r is None else tf.matmul(similarities, self.r)
        return logits if to_logits else tf.nn.sigmoid(logits)

    def loss(self, features: tf.Tensor):
        self._update_labels()
        regularize = 0 if self.regularize == 0 else (0.2*self.regularize/features.shape[0])*tf.reduce_sum(features*features)
        if self.similarity == "cos":
            features = tf.math.l2_normalize(features, axis=1)
        if self.loss_func == "diff":
            similarities = tf.multiply(tf.nn.embedding_lookup(features, self.edges[:, 0]), tf.nn.embedding_lookup(features, self.edges[:, 1]))
            logits = tf.reduce_sum(similarities, axis=1) if self.r is None else tf.matmul(similarities, self.r)
            return -tf.reduce_mean(tf.math.log_sigmoid(logits[0::2]-logits[1::2])) + regularize
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(self.labels, self.predict(features, to_logits=True)) + regularize

    def evaluate(self, features: tf.Tensor):
        self._update_labels()
        metric = tf.keras.metrics.AUC()
        metric.update_state(self.labels, self.predict(features))
        return metric.result().numpy()


class MeanLinkPrediction(LinkPrediction):
    def __init__(self, *args, graph, positive_nodes=None, negative_nodes=None, k=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_nodes = positive_nodes
        self.negative_nodes = negative_nodes
        self.k = k
        self.graph = graph
        self.parsed_edges = dict()
        for edge in self.edges:
            if edge[0] not in self.parsed_edges:
                self.parsed_edges[edge[0]] = list()
            if edge[1] not in self.parsed_edges:
                self.parsed_edges[edge[1]] = list()
            self.parsed_edges[edge[0]].append(edge[1])
            self.parsed_edges[edge[1]].append(edge[0])

    def evaluate(self, features):
        k = self.k
        aucs = list()
        precs = list()
        recs = list()
        f1s = list()
        maps = list()
        cov = set()

        positive_nodes = list(self.parsed_edges) if self.positive_nodes is None else self.positive_nodes
        negative_nodes = set([v for neighbors in self.parsed_edges for v in neighbors] if self.negative_nodes is None else self.negative_nodes)
        max_cov = set([neighbor for node in positive_nodes for neighbor in self.parsed_edges[node] if neighbor in negative_nodes])

        for node in tqdm(positive_nodes):
            if node not in self.parsed_edges:
                raise Exception("Node not found")
            node_positive_edges = [[node, neighbor] for neighbor in self.parsed_edges[node]]
            self.edges, self.labels = recommend_all(node,
                                               self.graph,
                                               positive_edges=node_positive_edges,
                                               negative_nodes=negative_nodes)
            prediction = self.predict(features).numpy()
            aucs.append(measures.auc(self.labels, prediction))
            maps.append(measures.avprec(self.labels, prediction, k))
            precs.append(measures.prec(self.labels, prediction, k))
            recs.append(measures.rec(self.labels, prediction, k))
            f1s.append(measures.f1(self.labels, prediction, k))
            #print(aucs[-1], precs[-1], recs[-1], f1s[-1], len(self.edges),len(node_positive_edges))
            cov = set(list(cov)+[self.edges[i][1] if self.edges[i][0] == node else self.edges[i][0] for i in np.argsort(prediction)[-k:]])
        print(f"Average node AUC {float(np.mean(aucs)):.3f}\t "
              f"MAP {float(np.mean(maps)):.3f}\t"
              f"Precision {float(np.mean(precs)):.3f}\t"
              f"Recall {float(np.mean(recs)):.3f}\t "
              f"F1 {float(np.mean(f1s)):.3f}\t "
              f"Coverage {float(len(cov) / len(max_cov)):.3f}")
        return np.mean(f1s)