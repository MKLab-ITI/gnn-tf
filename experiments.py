import tensorflow as tf
from gcn import ClassificationGCN, RelationalGCN
from experiment_setup import link_prediction_setup, classification_setup, semisupervised_classification_setup, graph2indices


class GeneralizedGCN(ClassificationGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_layer(self, layer_num, input_dims, output_dims):
        W = self.create_var(shape=(input_dims, output_dims), normalization='he')
        Wego = self.create_var(shape=(input_dims, output_dims), normalization='he')
        b = self.create_var(shape=(1, output_dims), normalization='zero')
        #activation = tf.nn.tanh if layer_num % 2 == 0 else tf.nn.softmax
        activation = tf.nn.tanh
        return W, Wego, b, activation

    def call_layer(self, layer, features):
        W, Wego, b, activation = layer
        features = tf.add(tf.sparse.sparse_dense_matmul(self.adjacency_matrix, tf.matmul(features, W)), tf.matmul(features, Wego)) + b
        #features = tf.sparse.sparse_dense_matmul(self.adjacency_matrix, tf.matmul(features, W))
        return activation(features)


class APPNP(ClassificationGCN):
    def __init__(self, *args, **kwargs):
        super(APPNP, self).__init__(*args, **kwargs)
        self.a = self.create_var(shape=(1, 1))

    def build_layer(self, layer_num, input_dims, output_dims):
        pass

    def preprocess(self, features):
        features = super().preprocess(features)
        self.H0 = features
        return features

    def call_layer(self, _, features):
        features = (1-self.a)*tf.sparse.sparse_dense_matmul(self.adjacency_matrix, features) + self.a*self.H0
        return features


def graph2adj(G):
    for u in G:
        G.add_edge(u, u)
    values = [(G.degree(u)*G.degree(v))**(-0.5) for u, v in G.edges()]
    return tf.sparse.SparseTensor(graph2indices(G), values, (len(G), len(G)))


acc = 0
repeats = 10
for _ in range(repeats):
    G, labels, training_idx, test_idx, node_features = semisupervised_classification_setup("cora")
    model = APPNP(graph2adj(G), node_features, output_labels=len(set(labels)), embedding_dims = [32]*32)
    acc += model.train(training_idx, labels[training_idx],
                validation_data=test_idx, validation_labels=labels[test_idx],
                learning_rate=0.01, epochs=250)
print('Overall accuracy', acc/repeats)

"""
G, edges, labels, training_idx, test_idx, node_features = link_prediction_setup("cora")
model = GeneralizedGCN(graph2adj(G), node_features)
model.train(edges[training_idx], labels[training_idx],
            validation_data=edges[test_idx], validation_labels=labels[test_idx],
            learning_rate=0.1, epochs=150)
"""