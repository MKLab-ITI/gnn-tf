import tensorflow as tf
from gnntf.core.nn import Trainable, Layer, Layered


class Structural(Layer):
    def __build__(self, architecture: Layered,
                  dims: int = 16,
                  l2_contraint: bool = False,
                  bipartite: int = 0,
                  **kwargs
                  ):
        top_shape = architecture.top_shape()
        self.l2_contraint = l2_contraint
        self.embeddings = architecture.create_var((bipartite, dims), **kwargs)
        self.embeddings2 = architecture.create_var((top_shape[0]-bipartite, dims), **kwargs)
        return top_shape[0], dims + top_shape[1]

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        embeddings = self.embeddings2
        if self.embeddings.shape[0] != 0:
            embeddings = tf.concat([self.embeddings, embeddings], axis=0)
        if self.l2_contraint:
            embeddings = tf.math.l2_normalize(embeddings, axis=1)
        if features.shape[0] == 0:
            return embeddings
        return tf.concat([embeddings, features], axis=1)


class GNN(Trainable):
    def __init__(self, graph: tf.Tensor, features: tf.Tensor, preprocessor: Layer = None):
        super().__init__(features)
        self.graph = graph
        if preprocessor is not None:
            self.add(preprocessor)

    def get_adjacency(self, graph_dropout=0.5, normalized="symmetric", add_eye="none"):
        graph = self.sparse_dropout(self.graph, graph_dropout)
        if add_eye == "before":
            graph = tf.sparse.add(graph, tf.sparse.eye(graph.shape[0]))
        if normalized == "symmetric":
            D = tf.math.divide_no_nan(1., tf.sqrt(tf.sparse.reduce_sum(graph, axis=0)))
            graph = tf.reshape(D, (-1, 1)) * graph * D
        elif normalized == "bipartite":
            D = tf.math.divide_no_nan(1., tf.sparse.reduce_sum(graph, axis=0))
            graph = tf.reshape(D, (-1, 1)) * graph
        elif normalized != "none":
            raise Exception("Invalid matrix normalization")
        if add_eye == "after":
            graph = tf.sparse.add(graph, tf.sparse.eye(graph.shape[0]))
        return graph