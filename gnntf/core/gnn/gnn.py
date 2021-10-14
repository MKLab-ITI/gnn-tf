import tensorflow as tf
from gnntf.core.nn import Trainable, Layer, Layered


class Structural(Layer):
    def __build__(self, architecture: Layered, structural_dims: int = 16, trainable: bool = True, normalization="xavier"):
        top_shape = architecture.top_shape()
        self.embeddings = architecture.create_var((top_shape[0], structural_dims), normalization, trainable=trainable)
        return top_shape[0], structural_dims + top_shape[1]

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        embeddings = tf.math.l2_normalize(self.embeddings, axis=1)
        if features.shape[0] == 0:
            return embeddings
        return tf.concat([embeddings, features], axis=1)


class GNN(Trainable):
    def __init__(self, G: tf.Tensor, features: tf.Tensor, structural_dims: int = 0):
        super().__init__(features)
        self.G = G
        if structural_dims != 0:
            self.add(Structural(structural_dims=structural_dims))

    def get_adjacency(self, graph_dropout=0.5, normalized=True, add_eye="before"):
        G = self.sparse_dropout(self.G, graph_dropout)
        if add_eye == "before":
            G = tf.sparse.add(G, tf.sparse.eye(G.shape[0]))
        if normalized:
            D = 1. / tf.sqrt(tf.sparse.reduce_sum(G, axis=0))
            G = tf.reshape(D, (-1, 1)) * G * D
        if add_eye == "after":
            G = tf.sparse.add(G, tf.sparse.eye(G.shape[0]))
        return G