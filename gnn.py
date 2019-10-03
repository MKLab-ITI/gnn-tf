from data.pairs import load
import tensorflow as tf
import networkx as nx
import numpy as np
import random
import sklearn.metrics


# tf.enable_eager_execution()


def negative_sampling(adjacency, max_negative_samples=500, node_ids=None):
    if node_ids is None:
        node_ids = list(range(adjacency.shape[0]))
    if max_negative_samples > len(node_ids):
        max_negative_samples = len(node_ids)
    training_edges = list()
    for i in node_ids:
        positives = np.sum(adjacency[i])
        negative_samples = int(max_negative_samples - positives)
        training_edges.extend([[i, j] for j in np.where(adjacency[i, :] == 1)[0]])
        if negative_samples>0:
            training_edges.extend([[i, j] for j in np.random.choice(np.where(adjacency[i, :] == 0)[0], negative_samples, replace=False)])
    return np.array(training_edges)


class Model:
    def __init__(self, input_dims, embedding_dims):
        self.vars = list()
        self.layers = list()
        for embedding_dim in embedding_dims:
            W = self._create_var(shape=(input_dims, embedding_dim))
            Wego = self._create_var(shape=(input_dims, embedding_dim))
            # b = self._create_var(shape=(1, embedding_dim))
            self.layers.append((W, Wego))
            input_dims = embedding_dim
        self.r = self._create_var(shape=(embedding_dim,1))
        self.training = True

    def _create_var(self, shape):
        var = tf.Variable(tf.random.normal(shape=shape))
        self.vars.append(var)
        return var

    @tf.function
    def _predict_batch(self, convolution, features, edges):
        for layer in self.layers:
            features = tf.add(tf.matmul(convolution, tf.matmul(features, layer[0])), tf.matmul(features, layer[1]))
            features = tf.nn.relu(features)
            if self.training:
                features = tf.keras.layers.Dropout(0.2)(features)
        #logits = tf.reduce_mean(tf.multiply(tf.gather(features, edges[:, 0]), tf.gather(features, edges[:, 1])), axis=1)
        logits = tf.matmul(tf.multiply(tf.gather(features, edges[:,0]), tf.gather(features, edges[:,1])), self.r)
        return tf.nn.sigmoid(logits)

    @tf.function
    def _train_batch(self, convolution, features, edges, labels, optimizer):
        with tf.GradientTape() as tape:
            prediction = self._predict_batch(convolution, features, edges)
            loss = -tf.reduce_sum(labels*tf.math.log(prediction+10E-6)) \
                    -tf.reduce_sum((1-labels)*tf.math.log(1-prediction+10E-6))
            regularizer = 0
            for var in self.vars:
                regularizer = regularizer + tf.nn.l2_loss(var)
            loss = loss + 0.01 * regularizer
        gradients = tape.gradient(loss, self.vars)
        optimizer.apply_gradients(zip(gradients, self.vars))
        return loss

    def train(self, convolution, features, edges, labels, batch_size=None, epochs=100, optimizer=None, tol=10E-6):
        self.training = True
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
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
            print('Epoch', epoch, '/', epochs, '\tLoss', loss)

    def evaluate(self, convolution, features, edges, labels):
        self.training = False
        prediction = self._predict_batch(convolution, features, edges)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prediction)
        return sklearn.metrics.auc(fpr, tpr)