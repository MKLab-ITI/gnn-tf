from data.nutella import load
import tensorflow as tf
import networkx as nx
import numpy as np
import sklearn.metrics


# tf.enable_eager_execution()


def negative_sampling(adjacency):
    training_edges = list()
    for i in range(adjacency.shape[0]):
        positives = np.sum(adjacency[i])
        remaining = adjacency.shape[0] // 20 - positives
        if remaining < 0:
            exit(1)
        for j in range(adjacency.shape[1]):
            if adjacency[i, j] == 1:
                training_edges.append([i, j])
            elif remaining > 0:
                training_edges.append([i, j])
                remaining -= 1
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

    def _create_var(self, shape, name=None):
        var = tf.Variable(tf.random.normal(shape=shape))
        self.vars.append(var)
        return var

    @tf.function
    def _predict_batch(self, convolution, features, edges):
        for layer in self.layers:
            features = tf.add(tf.matmul(convolution, tf.matmul(features, layer[0])), tf.matmul(features, layer[1])) #+ layer[2]
            features = tf.nn.relu(features)
            #features = tf.keras.layers.Dropout(0.2)(features)
        #logits = tf.reduce_mean(tf.multiply(tf.gather(features, edges[:, 0]), tf.gather(features, edges[:, 1])), axis=1)
        logits = tf.matmul(tf.multiply(tf.gather(features, edges[:,0]), tf.gather(features, edges[:,1])), self.r)
        return tf.nn.sigmoid(logits)

    @tf.function
    def _train_batch(self, convolution, features, edges, labels, optimizer):
        with tf.GradientTape() as tape:
            prediction = self._predict_batch(convolution, features, edges)
            loss = tf.reduce_sum(-tf.multiply(labels, tf.math.log(prediction+10E-6))-tf.multiply(1-labels, tf.math.log(1-prediction+10E-6)))
            regularizer = 0
            for var in self.vars:
                regularizer = regularizer + tf.nn.l2_loss(var)
            loss = loss + 0.01 * regularizer
        gradients = tape.gradient(loss, self.vars)
        optimizer.apply_gradients(zip(gradients, self.vars))
        return loss

    def train(self, convolution, features, edges, labels, batch_size=None, epochs=100, optimizer=None, tol=10E-6):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        if batch_size is None:
            batch_size = len(labels)
        prev_loss = float('inf')
        for epoch in range(epochs):
            print('Epoch', epoch, '/', epochs)
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
            print('\tLoss', loss)

    def evaluate(self, convolution, features, edges, labels, batch_size=None):
        if batch_size is None:
            batch_size = len(labels)
        prediction = self._predict_batch(convolution, features, edges)
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, prediction)
        return sklearn.metrics.auc(fpr, tpr)