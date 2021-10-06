import tensorflow as tf
import numpy as np
from gnntf.core.nn.layers import Layered
from gnntf.core.nn import Predictor
import random


class NodeClassification(Predictor):
    def __init__(self, nodes, labels=None, loss_transform=None):
        self.nodes = nodes
        self.labels = labels
        self.loss_transform = loss_transform

    def predict(self, features: tf.Tensor):
        return tf.argmax(tf.gather(features, self.nodes), axis=1) # data is a list of nodes

    def loss(self, features: tf.Tensor):
        if self.labels is None:
            raise Exception("Evaluation requires node labels")
        if self.loss_transform is not None:
            features = self.loss_transform(features)
        predictions = tf.nn.log_softmax(tf.gather(features, self.nodes), axis=1)
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(self.labels, predictions)

    def evaluate(self, features: tf.Tensor):
        if self.labels is None:
            raise Exception("Evaluation requires node labels")
        predictions = tf.argmax(tf.gather(features, self.nodes), axis=1)
        return 1-tf.math.count_nonzero(predictions-self.labels).numpy()/predictions.shape[0]


def negative_sampling(positive_edges, graph, samples=10):
    edges = list()
    values = list()
    nodes = list(graph)
    for u, v in positive_edges:
        edges.append([u, v])
        values.append(1)
        for _ in range(samples):
            vneg = v
            while graph.has_edge(u, vneg) or graph.has_edge(vneg, u) or vneg == u:
                vneg = random.choice(nodes)
            edges.append([u, vneg])
            values.append(0)
    return np.array(edges), values


class LinkPrediction(Predictor):
    def __init__(self, edges, labels=None, gnn: Layered=None):
        self.edges = edges
        self.values = None if labels is None else tf.constant(labels, shape=(len(labels),1))
        self.r = None if gnn is None else gnn.create_var(shape=(gnn.top_shape()[1],1), regularize=0, shared_name="distmult", normalization="ones", trainable=True)

    def predict(self, features: tf.Tensor, to_logits=False):
        features = tf.math.l2_normalize(features, axis=1)
        similarities = tf.multiply(tf.gather(features, self.edges[:,0]), tf.gather(features, self.edges[:,1]))
        logits = tf.reduce_sum(similarities, axis=1) if self.r is None else tf.matmul(similarities, self.r)
        return logits if to_logits else tf.nn.sigmoid(logits)

    def loss(self, features: tf.Tensor):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(self.values, self.predict(features, to_logits=True))

    def evaluate(self, features: tf.Tensor):
        metric = tf.keras.metrics.AUC()
        metric.update_state(self.values, self.predict(features))
        return metric.result().numpy()
