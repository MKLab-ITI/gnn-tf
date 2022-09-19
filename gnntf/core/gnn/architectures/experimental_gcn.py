from gnntf.core.gnn.gnn import GNN
from gnntf.core.gnn.architectures.gcn import GCNIILayer
from .experimental_filter import FastReg
import tensorflow as tf
from gnntf.core.nn.layers import Dense, Dropout



class GCNIIReg(GNN):
    # http://proceedings.mlr.press/v119/chen20v/chen20v.pdf
    def __init__(self, graph: tf.Tensor,
                 features: tf.Tensor,
                 num_classes,
                 a: float = 0.1,
                 l: float = 0.5,
                 latent_dims=[64],
                 iterations=64,
                 dropout = 0.6,
                 convolution_regularization=True,
                 **kwargs):
        super().__init__(graph, features, **kwargs)
        self.add(Dropout(dropout))
        for latent_dim in latent_dims:
            self.add(Dense(latent_dim, dropout=dropout, activation=tf.nn.relu))
        H0 = self.top_layer()
        self.add(FastReg())
        for iteration in range(iterations):
            self.add(GCNIILayer(H0, a, l, iteration, activation=tf.nn.relu, dropout=dropout, graph_dropout=0, regularization=convolution_regularization))
        self.add(Dense(num_classes, dropout=0, regularize=False))