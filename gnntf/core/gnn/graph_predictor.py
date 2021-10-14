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


def recommend_all(node, graph, positive_edges=None, negative_nodes=None):
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
        if v != node and not graph.has_edge(node, v) and not graph.has_edge(v, node):
            edges.append([node,v])
            values.append(0)
    return np.array(edges), values



def negative_sampling(positive_edges, graph, samples=1, negative_nodes=None):
    edges = list()
    values = list()
    if negative_nodes is None:
        negative_nodes = list(graph)
    for u, v in positive_edges:
        edges.append([u, v])
        values.append(1)
        for _ in range(samples):
            vneg = v
            while vneg == u or vneg == v or graph.has_edge(u, vneg) or graph.has_edge(vneg, u):
                vneg = random.choice(negative_nodes)
            edges.append([u, vneg])
            values.append(0)
    return np.array(edges), values


class LinkPrediction(Predictor):
    def __init__(self, edges, labels=None, gnn: Layered=None, similarity="cos", loss="ce"):
        if callable(edges):
            self.edge_sampler = edges
            edges, labels = edges()
        else:
            self.edge_sampler = None
        self.edges = edges
        self.loss_func = loss
        self.values = None if labels is None else tf.constant(labels, shape=(len(labels),1))
        self.r = None if gnn is None else gnn.create_var(shape=(gnn.top_shape()[1],1), regularize=0, shared_name="distmult", normalization="ones", trainable=True)
        self.similarity = similarity

    def _update_labels(self):
        if self.edge_sampler is not None:
            edges, labels = self.edge_sampler()
            self.edges = edges
            self.values = None if labels is None else tf.constant(labels, shape=(len(labels),1))

    def predict(self, features: tf.Tensor, to_logits=False):
        self._update_labels()
        if self.similarity == "cos":
            features = tf.math.l2_normalize(features, axis=1)
        similarities = tf.multiply(tf.nn.embedding_lookup(features, self.edges[:,0]), tf.nn.embedding_lookup(features, self.edges[:,1]))
        logits = tf.reduce_sum(similarities, axis=1) if self.r is None else tf.matmul(similarities, self.r)
        return logits if to_logits else tf.nn.sigmoid(logits)

    def loss(self, features: tf.Tensor):
        self._update_labels()
        #regularization = 1.5*tf.reduce_sum(tf.multiply(features, features))/len(self.edges)
        if self.loss_func == "diff":
            if self.similarity == "cos":
                features = tf.math.l2_normalize(features, axis=1)
            similarities = tf.multiply(tf.nn.embedding_lookup(features, self.edges[:, 0]), tf.nn.embedding_lookup(features, self.edges[:, 1]))
            logits = tf.reduce_sum(similarities, axis=1) if self.r is None else tf.matmul(similarities, self.r)
            return -tf.reduce_sum(tf.math.log_sigmoid(logits[0::2]-logits[1::2]))# + regularization
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(self.values, self.predict(features, to_logits=True))# + regularization

    def evaluate(self, features: tf.Tensor):
        self._update_labels()
        metric = tf.keras.metrics.AUC()
        metric.update_state(self.values, self.predict(features))
        return metric.result().numpy()


class MeanLinkPrediction(LinkPrediction):
    def __init__(self, *args, graph, positive_nodes=None, negative_nodes=None, k=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_nodes = positive_nodes
        self.negative_nodes = negative_nodes
        self.k = k
        self.graph = graph

    def evaluate(self, features):
        edges = self.edges
        graph = dict()
        for i, edge in enumerate(edges):
            if edge[0] not in graph:
                graph[edge[0]] = list()
            if edge[1] not in graph:
                graph[edge[1]] = list()
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        k = self.k
        aucs = list()
        precs = list()
        recs = list()
        f1s = list()
        maps = list()
        cov = set()
        max_cov = set()
        positive_nodes = list(graph) if self.positive_nodes is None else self.positive_nodes
        negative_nodes = list(graph) if self.negative_nodes is None else self.negative_nodes
        for node in tqdm(positive_nodes):
            if node not in graph:
                continue
            self.edges, labels = recommend_all(node, self.graph, [[node, neighbor] for neighbor in graph[node]], negative_nodes=negative_nodes)
            prediction = self.predict(features).numpy()
            aucs.append(measures.auc(labels, prediction))
            maps.append(measures.avprec(labels, prediction, k))
            precs.append(measures.prec(labels, prediction, k))
            recs.append(measures.rec(labels, prediction, k))
            f1s.append(measures.f1(labels, prediction, k))
            cov = set(list(cov)+[self.edges[i][1] if self.edges[i][0] == node else self.edges[i][0] for i in np.argsort(prediction)[-k:]])
            max_cov = set(list(cov)+[self.edges[i][1] if self.edges[i][0] == node else self.edges[i][0] for i in range(len(labels)) if labels[i]!=0])
        print(f"Average node AUC {float(np.mean(aucs)):.3f}\t "
              f"MAP {float(np.mean(maps)):.3f}\t"
              f"Precision {float(np.mean(precs)):.3f}\t"
              f"Recall {float(np.mean(recs)):.3f}\t "
              f"F1 {float(np.mean(f1s)):.3f}\t "
              f"Coverage {float(len(cov) / len(max_cov)):.3f}")
        self.edges = edges
        return np.mean(f1s)