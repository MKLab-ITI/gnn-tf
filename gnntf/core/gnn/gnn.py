import tensorflow as tf
from gnntf.core.nn import Trainable, Layer, Layered


class Positional(Layer):
    def __build__(self, architecture: Layered, positional_dims: int = 16, trainable: bool = True, normalization="bernouli"):
        top_shape = architecture.top_shape()
        self.embeddings = architecture.create_var((top_shape[0], positional_dims), normalization, trainable=trainable)
        return top_shape[0], positional_dims + top_shape[1]

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        if features.shape[0] == 0:
            return self.embeddings
        return tf.concat([self.embeddings, features], axis=1)


class GNN(Trainable):
    def __init__(self, G: tf.Tensor, features: tf.Tensor, positional_dims: int = 0):
        super().__init__(features)
        self.G = G
        if positional_dims != 0:
            self.add(Positional(positional_dims=positional_dims))

    def get_adjacency(self, graph_dropout=0.5, renormalized=True):
        G = self.sparse_dropout(self.G, graph_dropout)
        if renormalized:
            G = tf.sparse.add(G, tf.sparse.eye(G.shape[0]))
            D = 1. / tf.sqrt(tf.sparse.reduce_sum(G, axis=0))
            G = tf.reshape(D, (-1, 1)) * G * D
        return G