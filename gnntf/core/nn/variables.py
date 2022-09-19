import tensorflow as tf


class WrappedVariable(object):
    def __init__(self, shape, normalization='small', trainable=True, regularize=True, name=None):
        self.var = tf.Variable(tf.zeros(shape), trainable=trainable)
        self.trainable = trainable
        self.regularize = float(regularize)
        self.name = name
        self.normalization = normalization

    def apply_gradient(self, optimizer, gradient):
        if gradient is None:
            return
        optimizer.apply_gradients([(gradient, self.var)])

    def reset(self):
        if isinstance(self.normalization, float):
            self.var.assign(tf.keras.initializers.RandomUniform(-self.normalization, self.normalization)(self.var.shape))
        elif self.normalization == 'zero':
            self.var.assign(tf.zeros(self.var.shape))
        elif self.normalization == 'eye':
            self.var.assign(tf.eye(self.var.shape[1]))
        elif self.normalization == 'ones':
            self.var.assign(tf.ones(self.var.shape))
        elif self.normalization == 'xavier':
            self.var.assign(tf.keras.initializers.GlorotUniform()(self.var.shape))
        elif self.normalization == 'he':
            self.var.assign(tf.keras.initializers.HeUniform()(self.var.shape))
        elif self.normalization == 'bernouli':
            self.var.assign((tf.round(tf.random.uniform(self.var.shape))*2-1)/self.var.shape[1]**0.5)
        elif self.normalization == 'small':
            std = 1. / (self.var.shape[1] ** 0.5)
            self.var.assign(tf.keras.initializers.RandomUniform(-std, std)(self.var.shape))
        else:
            raise Exception("Invalid normalization type")

    def identity(self):
        return tf.identity(self.var)

    def numpy(self):
        return self.var.numpy()

    def assign(self, value):
        self.var.assign(value)


class VariableGenerator(object):
    def __init__(self):
        self.__vars = list()
        self.__named_vars = dict()

    def vars(self):
        return self.__vars

    def create_var(self, *args, shared_name=None, **kwargs):
        if shared_name is not None and shared_name in self.__named_vars:
            return self.__named_vars[shared_name]
        var = WrappedVariable(*args, **kwargs)
        self.__vars.append(var)
        if shared_name is not None:
            self.__named_vars[shared_name] = var.var
        return var.var

    def reset(self):
        for var in self.__vars:
            var.reset()