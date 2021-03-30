import tensorflow as tf
from core.nn.layers import Layered


class Predictor(object):
    def predict(self, features: tf.Tensor):
        raise Exception("Predictors need to implement a predict method")

    def loss(self, features: tf.Tensor):
        raise Exception("Predictors need to implement a loss method")

    def evaluate(self, features: tf.Tensor):
        raise Exception("Predictors need to implement an evaluate method")


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
        self.values = labels
        #self.r = gnn.create_var(shape=(gnn.top_shape()[1],1), shared_name="distmult")

    def predict(self, features: tf.Tensor):
        #logits = tf.matmul(tf.multiply(tf.gather(features, self.edges[:,0]), tf.gather(features, self.edges[:,1]), 1), self.r)
        #print(logits.shape)
        logits = tf.multiply(tf.gather(features, self.edges[:,0]), tf.gather(features, self.edges[:,1]), 1)
        logits = tf.reduce_sum(logits, 1)

        return tf.nn.sigmoid(logits)

    def loss(self, features: tf.Tensor):
        return tf.keras.losses.BinaryCrossentropy()(self.values, self.predict(features))

    def evaluate(self, features: tf.Tensor):
        metric = tf.keras.metrics.AUC()
        metric.update_state(self.values, self.predict(features))
        return metric.result().numpy()
