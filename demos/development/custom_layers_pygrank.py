import pygrank

from experiments.experiment_setup import dgl_setup
import gnntf
import tensorflow as tf
import pygrank as pg



G, labels, features, train, valid, test = dgl_setup("cora")
nxgraph = G
num_classes = len(set(labels))
pg.load_backend("tensorflow")


convergence = {"error_type": "iters", "max_iters": 10}
optimization = dict()
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric", renormalize=True)
algorithms = {
    "ppr0.5": pg.PageRank(alpha=0.5, preprocessor=pre, **convergence, use_quotient=False),
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, **convergence, use_quotient=False),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, **convergence, use_quotient=False),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, **convergence, use_quotient=False),
    "hk2": pg.HeatKernel(t=2, preprocessor=pre, **convergence, optimization_dict=optimization),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, **convergence, optimization_dict=optimization),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, **convergence, optimization_dict=optimization),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, **convergence, optimization_dict=optimization),
}


class PygrankLayer(gnntf.Layer):
    def __build__(self, architecture: gnntf.GNN, restart_probability: float = 0.1,
                  activation=lambda x: x, dropout: float = 0, graph_dropout: float = 0.5, restart_transform=lambda x: x):
        self.restart_probability = restart_probability
        self.dropout = dropout
        self.graph_dropout = graph_dropout
        self.activation = activation
        self.restart_transform = restart_transform
        self.algorithm = pg.PageRank(0.9, assume_immutability=True, max_iters=10, error_type="iters",
                                     normalization="symmetric", renormalize=True, use_quotient=False)
        self.algorithm = pg.ParameterTuner(lambda params:
            pg.PageRank(params[0], assume_immutability=True, max_iters=10, error_type="iters",
                        normalization="symmetric", renormalize=True, use_quotient=False),
            max_vals=[0.99], min_vals=[.5], divide_range=2, measure=pg.KLDivergence,
                                           deviation_tol=0.1, tuning_backend="numpy", fraction_of_training=.2)
        return architecture.top_shape() # preserves the feature shape

    def __forward__(self, architecture: gnntf.GNN, features: tf.Tensor):
        #self.G = architecture.get_adjacency(self.graph_dropout)
        activation = self.algorithm.propagate(nxgraph, features, graph_dropout=self.graph_dropout)
        return self.activation(architecture.dropout(activation, self.dropout))


class PygrankAPPNP(gnntf.GNN):
    # https://arxiv.org/pdf/1810.05997.pdf
    def __init__(self, G: tf.Tensor, features: tf.Tensor, num_classes: int, a: float = 0.1, latent_dims=[64], iterations=10,
                 dropout = 0.6, graph_dropout=0.5, activation=lambda x: x, **kwargs):
        super().__init__(G, features, **kwargs)
        for latent_dim in latent_dims:
            self.add(gnntf.Dense(latent_dim, activation=tf.nn.relu, dropout=dropout))
        self.add(gnntf.Dense(num_classes, regularize=False))
        self.add(PygrankLayer(a, graph_dropout=graph_dropout, activation=activation))

pygrank.load_backend("tensorflow")
gnn = PygrankAPPNP(gnntf.graph2adj(G), features, num_classes)

gnn.train(train=gnntf.NodeClassification(train, labels[train]),
          valid=gnntf.NodeClassification(valid, labels[valid]),
          test=gnntf.NodeClassification(test, labels[test]),
          verbose=True)

prediction = gnn.predict(gnntf.NodeClassification(test))
accuracy = gnntf.acc(prediction, labels[test])
print(accuracy)
