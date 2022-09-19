from gnntf.core.nn import Layer, Dense, Activation
from gnntf.core.gnn.gnn import GNN
from gnntf.core.gnn.architectures.filter import PPRIteration
import tensorflow as tf


class PPRSweep(Layer):
    def __build__(self, architecture: GNN, restart_probability: float = 0.1):
        self.restart_probability = restart_probability
        return architecture.top_shape() # preserves the feature shape

    def __forward__(self, architecture: GNN, features: tf.Tensor):
        self.G = architecture.get_adjacency()
        H0 = features*0 + 1
        HN = H0
        for _ in range(10):
            HN = tf.sparse.sparse_dense_matmul(self.G, HN)*(1-self.restart_probability) + H0*self.restart_probability

        return features/HN


class FastReg(Layer):
    def __build__(self, architecture: GNN):
        self.output_regularize = 1
        self.internal_shape = (architecture.top_shape()[1], 1)
        return architecture.top_shape()

    def __forward__(self, architecture: GNN, features: tf.Tensor):
        self.features = features
        self.architecture = architecture
        self.W = architecture.create_var(self.internal_shape, regularize=1)
        return features

    def loss(self):
        G = self.architecture.get_adjacency(normalized="none")
        features = tf.nn.sigmoid(tf.matmul(self.features, self.W))
        propagated = tf.sparse.sparse_dense_matmul(G, features)
        diffs = features-propagated
        D = tf.sparse.reduce_sum(G, axis=0)
        #graph = tf.reshape(D, (-1, 1))
        lam = tf.reduce_sum(diffs*diffs)/tf.reduce_sum(tf.reshape(D, (-1, 1))*features*features)
        #print(lam)
        return -lam


class APPNPReg(GNN):
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes: int, a: float = 0.1, latent_dims=[64], iterations=10,
                 dropout=0.6, graph_dropout=0.5, activation=lambda x: x, **kwargs):
        super().__init__(G, features, **kwargs)
        for latent_dim in latent_dims:
            self.add(Dense(latent_dim, activation=tf.nn.relu, dropout=dropout))
        H0 = self.add(Dense(num_classes, regularize=False))
        #self.add(FastReg())
        for _ in range(iterations):
            self.add(PPRIteration(H0, self.create_var() if a is None else a, graph_dropout=graph_dropout, activation=activation))
