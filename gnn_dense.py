import tensorflow as tf
import numpy as np
import networkx as nx


# tf.enable_eager_execution()


def get_adjacency(G):
    return np.array(nx.to_numpy_matrix(G)).astype('float32')


def get_labels(adjacency, edges, expand=True):
    result = np.array([adjacency[i, j] for i,j in edges])
    if expand:
        result = np.expand_dims(result, axis=1)
    return result.astype('float32')


def get_convolution(adjacency, remove_edges=None):
    convolution = np.copy(adjacency)
    if remove_edges is not None:
        for i, j in remove_edges:
            convolution[i, j] = 0
            convolution[j, i] = 0
    return convolution / (np.sum(convolution, axis=1)+np.finfo(float).eps)


def negative_sampling(adjacency, max_negative_samples=100, node_ids=None):
    if node_ids is None:
        node_ids = list(range(adjacency.shape[0]))
    if max_negative_samples > len(node_ids):
        max_negative_samples = len(node_ids)
    training_edges = list()
    for i in node_ids:
        positives = np.sum(adjacency[i])
        negative_samples = int(max_negative_samples - positives)
        training_edges.extend([[i, j] for j in np.where(adjacency[i, :] == 1)[0]])
        if negative_samples > 0:
            training_edges.extend([[i, j] for j in np.random.choice(np.where(adjacency[i, :] == 0)[0], negative_samples, replace=False) if i!=j])
    return np.array(training_edges)


class Random:
    def __init__(self):
        pass

    def train(self, convolution, features, training_edges, training_labels, epochs=50):
        pass

    def predict(self, convolution, features, edges):
        return [np.random.choice([0,1]) for i,j in edges]

class Model:
    def __init__(self, input_dims, embedding_dims, activation="none"):
        self.vars = list()
        self.layers = list()
        for embedding_dim in embedding_dims:
            W = self._create_var(shape=(input_dims, embedding_dim))
            Wego = self._create_var(shape=(input_dims, embedding_dim))
            b = self._create_var(shape=(1, embedding_dim))
            self.layers.append((W, Wego, b))
            input_dims = embedding_dim
        self.r = self._create_var(shape=(embedding_dim,1))
        self.training = True
        self.activation = activation

    def _create_var(self, shape):
        var = tf.Variable(tf.random.normal(shape=shape))
        #var = tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=shape))
        self.vars.append(var)
        return var

    def _predict_batch(self, convolution, features, edges):
        for layer in self.layers:
            features = tf.add(tf.matmul(convolution, tf.matmul(features, layer[0])), tf.matmul(features, layer[1])) + layer[2]
            if "relu" in self.activation:
                features = tf.nn.relu(features)
            if "sigmoid" in self.activation:
                features = tf.nn.sigmoid(features)
            if "tanh" in self.activation:
                features = tf.nn.tanh(features)
            if self.training:
                features = tf.keras.layers.Dropout(0.3)(features)
        logits = tf.matmul(tf.multiply(tf.gather(features, edges[:,0]), tf.gather(features, edges[:,1])), self.r)
        return tf.nn.sigmoid(logits)

        #diffs = tf.gather(features, edges[:,0])-tf.gather(features, edges[:,1])
        #logits = tf.matmul(tf.multiply(diffs, diffs), self.r)
        #return 1/(1+logits)

    def _train_batch(self, convolution, features, edges, labels, optimizer):
        with tf.GradientTape() as tape:
            prediction = self._predict_batch(convolution, features, edges)
            loss = -tf.reduce_sum(labels*tf.math.log(prediction+1.E-9)) \
                    -tf.reduce_sum((1-labels)*tf.math.log(1-prediction+1.E-9))
            regularizer = 0
            for var in self.vars:
                regularizer = regularizer + tf.nn.l2_loss(var) / len(self.layers)
            loss = loss + 0.01 * regularizer
        gradients = tape.gradient(loss, self.vars)
        optimizer.apply_gradients(zip(gradients, self.vars))
        return loss

    def train(self, convolution, features, edges, labels, batch_size=None, epochs=100, optimizer=None, tol=10E-6):
        self.training = True
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        if batch_size is None:
            batch_size = len(labels)
        prev_loss = float('inf')
        for epoch in range(epochs):
            batch_start = 0
            loss = 0
            while batch_start<len(labels):
                batch_edges = edges[batch_start:min(batch_start+batch_size, len(labels)), :]
                batch_labels = labels[batch_start:min(batch_start+batch_size, len(labels))]
                loss += self._train_batch(convolution, features, batch_edges, batch_labels, optimizer)
                batch_start += batch_size
            loss /= len(labels)
            if abs(loss-prev_loss)<tol:
                break
            prev_loss = loss
            #print('Epoch', epoch, '/', epochs, '\tLoss', loss)

    def predict(self, convolution, features, edges):
        self.training = False
        return self._predict_batch(convolution, features, edges)



class PageRank:
    def __init__(self, alpha=0.85, layers=None):
        if layers is None:
            layers = int(2.0/(1-alpha))
        self.layers = layers
        self.alpha = alpha
        self.trained = None

    @tf.function
    def _predict_batch(self, convolution, features):
        personalization = features
        for layer in range(self.layers):
            features = tf.matmul(convolution, features)*self.alpha + (1-self.alpha)*personalization
        #logits = tf.reduce_mean(tf.multiply(tf.gather(features, edges[:, 0]), tf.gather(features, edges[:, 1])), axis=1)
        return features

    def train(self, convolution, features, training_edges, training_labels, epochs=None):
        self.features = self._predict_batch(convolution, features)

    def predict(self, convolution, features, edges):
        return [self.features[i,j] for i,j in edges]