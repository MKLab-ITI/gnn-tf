import tensorflow as tf
from core.nn.trainable import Trainable


class GNN(Trainable):
    def __init__(self, G: tf.Tensor, features: tf.Tensor):
        super().__init__(features)
        self.G = G

    def get_adjacency(self, graph_dropout=0.5, renormalized=True):
        G = self.sparse_dropout(self.G, graph_dropout)
        if renormalized:
            G = tf.sparse.add(G, tf.sparse.eye(G.shape[0]))
            D = 1. / tf.sqrt(tf.sparse.reduce_sum(G, axis=0))
            G = tf.reshape(D, (-1, 1)) * G * D
        return G