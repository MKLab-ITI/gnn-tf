from .layered import Layered, Layer
import tensorflow as tf


class Branch(Layer):
    def __build__(self, architecture: Layered, features: tf.Tensor):
        self.features = features
        return self.features.shape

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        return self.features


class Resume(Layer):
    def __build__(self, architecture: Layered, H0: Layer):
        self.H0 = H0
        return H0.output_shape

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        return self.H0.value


class Concatenate(Layer):
    def __build__(self, architecture: Layered, H0: Layer):
        if architecture.top_shape()[0] != H0.output_shape[0]:
            raise Exception("Mismatching first dimension to concatenate between shapes "+str(architecture.top_shape())+" and "+str(H0.output_shape))
        self.H0 = H0
        return (architecture.top_shape()[0], architecture.top_shape()[1]+H0.output_shape[1])

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        ret = tf.concat([features, self.H0.value], 1)
        return ret
    

class Tradeoff(Layer):
    def __build__(self, architecture: Layered, layers, weights=None, trainable=True):
        shape = layers[0].output_shape
        for layer in layers:
            if layer.output_shape!= shape:
                raise Exception("Mismatching trade-off dimentions")
        self.layers = layers
        self.weights = [architecture.create_var((1,1), "zero", trainable=trainable) for _ in layers] if weights is None else weights
        return shape

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        ret = 0
        weight_sum = 0
        for weight in self.weights:
            weight_sum = weight_sum + tf.sigmoid(weight)
        for weight, layer in zip(self.weights, self.layers):
            ret = ret + tf.sigmoid(weight)*layer.value / weight_sum
        print([float(tf.sigmoid(weight).numpy()) for weight in self.weights])
        return ret


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
        elif activation == "kernel":
            scale1 = architecture.create_var((1, 1), "ones", regularize=False)
            scale2 = architecture.create_var((1, 1), "zero", regularize=False)
            scale3 = architecture.create_var((1, 1), "zero", regularize=False)
            scale4 = architecture.create_var((1, 1), "zero", regularize=False)
            scale5 = architecture.create_var((1, 1), "zero", regularize=False)
            scale6 = architecture.create_var((1, 1), "zero", regularize=False)
            activation = lambda x: tf.math.log(tf.exp(x*scale1+scale4) + tf.exp(x*scale2+scale5) + tf.exp(x*scale3+scale6))
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
