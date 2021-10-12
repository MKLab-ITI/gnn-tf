from gnntf.core.nn import Layer, Layered, Concatenate
from gnntf.core.gnn.gnn import GNN
import tensorflow as tf
from gnntf.core.nn.layers import Dense, Dropout


class GCNIILayer(Layer):
    def __build__(self, architecture: Layered, H0: Layer, a: float, l: float, k: int = 0,
                  activation=lambda x: x, beta_transformer=tf.math.log1p,
                  dropout: float = 0.5, graph_dropout: float = 0.5, regularization=True):
        self.W = architecture.create_var((architecture.top_shape()[1], architecture.top_shape()[1]), "zero", regularize=regularization)
        self.a = a
        self.l = l
        self.k = k
        self.activation = activation
        self.dropout = dropout
        self.graph_dropout = graph_dropout
        self.H0 = H0
        self.beta_transformer = beta_transformer
        return architecture.top_shape()

    def __forward__(self, gcn, features: tf.Tensor):
        b = self.beta_transformer(self.l / (self.k+1))
        aggregated_features = tf.sparse.sparse_dense_matmul(gcn.get_adjacency(self.graph_dropout), features)
        tradeoff = (1-self.a)*aggregated_features + self.a * self.H0.value
        activation = tf.matmul(tradeoff, (1 - b) * tf.eye(self.W.shape[1]) + b * self.W)
        return gcn.dropout(self.activation(activation), self.dropout)


class GCNII(GNN):
    # http://proceedings.mlr.press/v119/chen20v/chen20v.pdf
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes, a = 0.1, l=0.5, latent_dims=[64], iterations=64, dropout = 0.6, convolution_regularization=True, **kwargs):
        super().__init__(G, features, **kwargs)
        self.add(Dropout(dropout))
        for latent_dim in latent_dims:
            self.add(Dense(latent_dim, dropout=dropout, activation=tf.nn.relu))
        H0 = self.top_layer()
        for iteration in range(iterations):
            self.add(GCNIILayer(H0, a, l, iteration, activation=tf.nn.relu, dropout=dropout, graph_dropout=0, regularization=convolution_regularization))
        self.add(Dense(num_classes, dropout=0, regularize=False))


class GCNLayer(Layer):
    def __build__(self, gcn, outputs: int, activation=tf.nn.relu, bias: bool=True,
                  dropout: float=0, graph_dropout: float=0):
        self.W = gcn.create_var((gcn.top_shape()[1], outputs))
        self.b = gcn.create_var((1,outputs), "zero") if bias else 0
        self.activation = activation
        self.dropout = dropout
        self.graph_dropout = graph_dropout
        return (gcn.top_shape()[0], outputs)

    def __forward__(self, gcn, features: tf.Tensor):
        aggregated_features = tf.sparse.sparse_dense_matmul(gcn.get_adjacency(self.graph_dropout), features)
        return gcn.dropout(self.activation(tf.matmul(aggregated_features, self.W) + self.b), self.dropout)


class GCN(GNN):
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes, latent_dims=[64], **kwargs):
        super().__init__(G, features, **kwargs)
        for latent_dim in latent_dims:
            self.add(GCNLayer(latent_dim, graph_dropout=0.5))
            self.add(Dropout(0.5))
        self.add(GCNLayer(num_classes))


class GRecLayer(Layer):
    def __build__(self, gcn, outputs: int, activation=tf.nn.leaky_relu, bias: bool=False,
                  dropout: float = 0, node_dropout: float = 0, regularize: float = 1):
        self.W1 = gcn.create_var((gcn.top_shape()[1], outputs), regularize=regularize)
        self.W2 = gcn.create_var((gcn.top_shape()[1], outputs), regularize=regularize)
        self.b = gcn.create_var((1,outputs), "zero") if bias else 0
        self.activation = activation
        self.dropout = dropout
        self.node_dropout = node_dropout
        return (gcn.top_shape()[0], outputs)

    def __forward__(self, gcn, features: tf.Tensor):
        aggregated_features = tf.sparse.sparse_dense_matmul(gcn.get_adjacency(self.node_dropout, add_eye="after"), features)
        neighbor_features = tf.sparse.sparse_dense_matmul(gcn.get_adjacency(self.node_dropout, add_eye="none"), features)
        neighbor_features = tf.multiply(neighbor_features, features)
        return gcn.dropout(self.activation(tf.matmul(aggregated_features, self.W1)
                                           +tf.matmul(neighbor_features, self.W2) + self.b), self.dropout)


class GRec(GNN):
    # https://dl.acm.org/doi/pdf/10.1145/3468264.3468552
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes: int, latent_dims=None, node_dropout=0, dropout=0.1, **kwargs):
        super().__init__(G, features, **kwargs)
        if latent_dims is None:
            latent_dims = [num_classes]*2
        layers = list()
        self.add(Dropout(dropout))
        for latent_dim in latent_dims:
            layers.append(self.add(GRecLayer(latent_dim, node_dropout=node_dropout, regularize=0.2)))
            self.add(Dropout(dropout))
        layers.append(self.add(GRecLayer(num_classes, regularize=20.)))
        self.add(Concatenate(layers))

