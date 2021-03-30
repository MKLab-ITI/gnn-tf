from core.nn.layered import Layer
from core.gnn.gnn import GNN
import tensorflow as tf
from core.nn.layers import Dense, Branch, Concatenate


class PPRIteration(Layer):
    def __build__(self, architecture: GNN, H0: Layer, restart_probability: float = 0.1,
                  activation=lambda x: x, dropout: float = 0, graph_dropout: float = 0.5, restart_transform=lambda x: x):
        self.restart_probability = restart_probability
        self.H0 = H0
        self.dropout = dropout
        self.graph_dropout = graph_dropout
        self.activation = activation
        self.restart_transform = restart_transform
        return architecture.top_shape() # preserves the feature shape

    def __forward__(self, architecture: GNN, features: tf.Tensor):
        self.G = architecture.get_adjacency(self.graph_dropout)
        propagated = tf.sparse.sparse_dense_matmul(self.G, features)
        a = self.restart_transform(self.restart_probability)
        activation = propagated*(1-a) + self.H0.value*a
        return self.activation(architecture.dropout(activation, self.dropout))


class StableIteration(Layer):
    def __build__(self, architecture: GNN, H0: Layer, a: float = 0.1,
                  activation=lambda x: x, dropout: float = 0, graph_dropout: float = 0.5, restart_transform=lambda x: x):
        self.a = a
        self.H0 = H0
        self.dropout = dropout
        self.graph_dropout = graph_dropout
        self.activation = activation
        self.restart_transform = restart_transform
        return architecture.top_shape() # preserves the feature shape

    def __forward__(self, architecture: GNN, features: tf.Tensor):
        self.G = architecture.get_adjacency(self.graph_dropout)
        propagated = tf.sparse.sparse_dense_matmul(self.G, features)
        a = self.restart_transform(self.a)
        activation = propagated*a + self.H0.value
        return self.activation(architecture.dropout(activation, self.dropout))


class APPNP(GNN):
    # https://arxiv.org/pdf/1810.05997.pdf
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes: int, a: float = 0.1, latent_dims=[64], iterations=10,
                 dropout = 0.6, graph_dropout=0.5, activation=lambda x: x, enable_error: bool = False, **kwargs):
        super().__init__(G, features, **kwargs)
        for latent_dim in latent_dims:
            self.add(Dense(latent_dim, activation=tf.nn.relu, dropout=dropout))
        H0 = self.add(Dense(num_classes, regularize=False))
        if enable_error:
            self.Hlabel = Branch(tf.zeros((features.shape[0], 1)))
            H0 = self.add(Concatenate(H0))
        else:
            self.Hlabel = None
        for _ in range(iterations):
            self.add(PPRIteration(H0, a, graph_dropout=graph_dropout, activation=activation))
        if enable_error:
            self.add(Dense(num_classes))

    def train(self, train, *args, **kwargs):
        if self.Hlabel is not None:
            self.Hlabel = train.nodes
        return super().train(train, *args, **kwargs)


class APExp(GNN):
    # https://arxiv.org/pdf/1810.05997.pdf
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes: int, a: float = 1., latent_dims=[64], iterations=10,
                 dropout = 0.6, graph_dropout=0.5, activation=lambda x: x, **kwargs):
        super().__init__(G, features, **kwargs)
        for latent_dim in latent_dims:
            self.add(Dense(latent_dim, activation=tf.nn.relu, dropout=dropout))
        H0 = self.add(Dense(num_classes, regularize=False))
        for iteration in range(iterations):
            self.add(PPRIteration(H0, a/(1+iteration), graph_dropout=graph_dropout, activation=activation))