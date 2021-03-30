from .variables import VariableGenerator
import tensorflow as tf


class Layered(VariableGenerator):
    def __init__(self, input_shape, layers=list()):
        super().__init__()
        self.__layers = list()
        self.__training_mode = True
        self.input_shape = input_shape
        for layer in layers:
            self.add(layer)

    def layers(self):
        return self.__layers

    def top_shape(self):
        if not self.__layers:
            return self.input_shape
        return self.__layers[-1].output_shape

    def top_layer(self):
        return self.__layers[-1]

    def add(self, layer):
        if layer not in self.__layers:
            layer.__late_init__(self)
        self.__layers.append(layer)
        return layer

    def training_mode(self, training_mode):
        self.__training_mode = training_mode

    def __enter__(self):
        self.__training_mode = True
        return [var.var for var in self.vars() if var.trainable]

    def __exit__(self, type, value, tb):
        self.__training_mode = False

    def dropout(self, features, dropout=0.5):
        return tf.nn.dropout(features, dropout) if self.__training_mode and dropout != 0 else features

    def sparse_dropout(self, G, dropout=0.5):
        return tf.SparseTensor(G.indices, tf.nn.dropout(G.values, dropout), G.dense_shape) if self.__training_mode and dropout != 0 else G

    def __call__(self, features: tf.Tensor):
        for layer in self.__layers:
            features = layer(self, features)
        return features


class Layer(object):
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def __late_init__(self, architecture: VariableGenerator):
        prev_vars = set(architecture.vars())
        self.output_shape = self.__build__(architecture, *self.__args, **self.__kwargs)
        if self.output_shape is None:
            raise Exception("Layer __build__ should return an output shape")
        self.vars = set(architecture.vars())-prev_vars
        self.__args = None
        self.__kwargs = None

    def __build__(self, architecture: VariableGenerator, *args, **kwargs):
        raise Exception("Layer should implment a __build__ method")

    def __forward__(self, architecture: VariableGenerator, features: tf.Tensor):
        raise Exception("Layer should implement a __forward__ method")

    def __call__(self, architecture: VariableGenerator, features: tf.Tensor):
        self.value = self.__forward__(architecture, features)
        return self.value