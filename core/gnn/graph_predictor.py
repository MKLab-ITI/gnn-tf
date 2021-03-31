import tensorflow as tf
from core.nn.layers import Layered
from core.nn.trainable import Predictor


class NodeClassification(Predictor):
    def __init__(self, nodes, labels=None, loss_transform=None):
        self.nodes = nodes
        self.labels = labels
        self.loss_transform = loss_transform

    def predict(self, features: tf.Tensor):
        return tf.argmax(tf.gather(features, self.nodes), axis=1) # data is a list of nodes

    def loss(self, features: tf.Tensor):
        if self.labels is None:
            raise Exception("Evaluation requires node labels")
        if self.loss_transform is not None:
            features = self.loss_transform(features)
        predictions = tf.nn.log_softmax(tf.gather(features, self.nodes), axis=1)
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(self.labels, predictions)

    def evaluate(self, features: tf.Tensor):
        if self.labels is None:
            raise Exception("Evaluation requires node labels")
        predictions = tf.argmax(tf.gather(features, self.nodes), axis=1)
        return 1-tf.math.count_nonzero(predictions-self.labels).numpy()/predictions.shape[0]


class LinkPrediction(Predictor):
    def __init__(self, edges, labels, gnn : Layered):
        self.edges = edges
        self.values = tf.constant(labels, shape=(len(labels),1))
        self.r = gnn.create_var(shape=(gnn.top_shape()[1],1), regularize=0, shared_name="distmult", normalization="ones")

    def predict(self, features: tf.Tensor, to_logits=False):
        #features = tf.math.l2_normalize(features, axis=1)
        logits = tf.matmul(tf.multiply(tf.gather(features, self.edges[:,0]), tf.gather(features, self.edges[:,1])), self.r)
        if not to_logits:
            logits = tf.nn.sigmoid(logits)
        return logits
        #logits = tf.multiply(tf.gather(features, self.edges[:,0]), tf.gather(features, self.edges[:,1]))
        #return tf.nn.sigmoid(tf.reduce_sum(logits, 1))

    def loss(self, features: tf.Tensor):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(self.values, self.predict(features, to_logits=True))

    def evaluate(self, features: tf.Tensor):
        #pred = self.predict(features)
        #print(pred.numpy()[0:30])
        #print(self.values[0:30])
        metric = tf.keras.metrics.AUC()
        metric.update_state(self.values, self.predict(features))
        return metric.result().numpy()
