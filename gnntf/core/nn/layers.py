from .layered import Layered, Layer
import tensorflow as tf


class LSTM(Layer):
    def __build__(self, architecture: Layered, dims, dict_size):
        self.Wf = architecture.create_var((dims, dims), regularize=100)
        self.Uf = architecture.create_var((dims, dims), regularize=100)
        self.bf = architecture.create_var((1, dims), "zero", regularize=False)
        self.Wi = architecture.create_var((dims, dims), regularize=100)
        self.Ui = architecture.create_var((dims, dims), regularize=100)
        self.bi = architecture.create_var((1, dims), "zero", regularize=False)
        self.Wo = architecture.create_var((dims, dims), regularize=100)
        self.Uo = architecture.create_var((dims, dims), regularize=100)
        self.bo = architecture.create_var((1, dims), "zero", regularize=False)
        self.Wc = architecture.create_var((dims, dims), regularize=100)
        self.Uc = architecture.create_var((dims, dims), regularize=100)
        self.bc = architecture.create_var((1, dims), "zero", regularize=False)
        self.embeddings = architecture.create_var((dict_size, dims))
        return (architecture.top_shape()[0], dims*2)

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        def mul(W, r):
            if tf.reduce_sum(r) == 0:
                return r
            return tf.matmul(r, W)
        c = 0
        h = 0
        features = features.numpy().astype(int)
        for t in range(features.shape[1]):
            xt = tf.gather(self.embeddings, features[:, t], axis=0)
            ft = tf.nn.sigmoid(mul(self.Wf, xt) + mul(self.Uf, h) + self.bf)
            ot = tf.nn.sigmoid(mul(self.Wo, xt) + mul(self.Uo, h) + self.bo)
            it = tf.nn.sigmoid(mul(self.Wi, xt) + mul(self.Ui, h) + self.bi)
            ct = tf.nn.tanh(mul(self.Wc, xt) + mul(self.Uc, h) + self.bc)
            c = ft*c + it*ct
            h = ot*tf.nn.tanh(c)
        ret = tf.concat([h, c], axis=1)
        return ret

    def loss(self):
        return 0


class Wrap(Layer):
    def __build__(self, architecture: Layered, tf_layer, *args, dropout=0, **kwargs):
        self.tf_layer = tf.keras.models.Sequential([tf.keras.Input(shape=(None, architecture.top_shape()[1])),
                                                                  tf_layer(*args, **kwargs)])
        def nop():
            pass
        for var in self.tf_layer.weights:
            architecture.create_var((1,1), regularize=False)
            architecture.vars()[-1].var = var
            architecture.vars()[-1].reset = nop
        self.dropout = dropout
        return (architecture.top_shape()[0], self.tf_layer.output_shape[-1])

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        return architecture.dropout(self.tf_layer(features), self.dropout)

    def loss(self):
        loss = super().loss()
        for layer in self.tf_layer.layers:
            loss = loss + tf.math.reduce_sum(layer.losses)
        return loss


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
        self.H0 = H0
        if isinstance(H0, list):
            for H in H0:
                if architecture.top_shape()[0] != H.output_shape[0]:
                    raise Exception("Mismatching first dimension to concatenate between shapes "+str(architecture.top_shape())+" and "+str(H0.output_shape))
            return (architecture.top_shape()[0], architecture.top_shape()[1]+H0[0].output_shape[1])
        if architecture.top_shape()[0] != H0.output_shape[0]:
            raise Exception("Mismatching first dimension to concatenate between shapes "+str(architecture.top_shape())+" and "+str(H0.output_shape))
        return (architecture.top_shape()[0], architecture.top_shape()[1]+H0.output_shape[1])

    def __forward__(self, architecture: Layered, features: tf.Tensor):
        if isinstance(self.H0, list):
            return tf.concat([H.value for H in self.H0], axis=0)
        return tf.concat([features, self.H0.value], axis=0)
    

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
    def __build__(self, architecture: Layered, outputs: int = None, activation = lambda x: x, bias: bool = True, dropout: float = 0, regularize : bool = True):
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
