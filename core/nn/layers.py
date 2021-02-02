from .layered import Layered, Layer
import tensorflow as tf


class Dense(Layer):
    def __build__(self, architecture: Layered, outputs: int = None, activation = lambda x: x, bias: bool = True, dropout: float = 0.5, regularize : bool = True):
        if outputs is None:
            outputs = architecture.top_shape()[1]
        self.W = architecture.create_var((architecture.top_shape()[1], outputs), regularize=regularize)
        self.b = architecture.create_var((1,outputs), "zero", regularize=regularize) if bias else 0
        self.activation = activation
        self.dropout = dropout
        return (architecture.top_shape()[0], outputs)

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        return architecture.dropout(self.activation(tf.matmul(features, self.W) + self.b), self.dropout)


class Activation(Layer):
    def __build__(self, architecture: Layered, activation: str = "relu", **kwargs):
        if activation == "relu":
            activation = tf.nn.relu
        elif activation == "linear":
            activation = lambda x: x
        elif activation == "tanh":
            activation = tf.nn.tanh
        elif activation == "exp":
            activation = tf.exp
        elif activation == "softmax":
            activation = lambda x: tf.nn.softmax(x, axis=1)
        elif activation == "scale":
            scale = architecture.create_var((1, 1), "zero", regularize=False)
            activation = lambda x: x * (1 + scale)
        elif activation == "softthresh":
            if 'threshold' in kwargs:
                theta = kwargs['threshold']
            else:
                theta = architecture.create_var((1, 1), "zero", regularize=False)
            activation = lambda x: tf.nn.relu(x - theta) - tf.nn.relu(theta - x)
        self.activation = activation
        return architecture.top_shape()

    def __forward__(self, gcn, features: tf.Tensor):
        return self.activation(features)


class Dropout(Layer):
    def __build__(self, gcn, rate: float = 0.5):
        self.rate = rate
        return gcn.top_shape()

    def __forward__(self, gcn, features: tf.Tensor):
        return gcn.dropout(features, self.rate)
