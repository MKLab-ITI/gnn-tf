import tensorflow as tf
from core.nn.layered import Layered
from core.nn.batching import batches


class _NodePredictor(object):
    def __init__(self, data):
        self.data = data

    def __call__(self, features):
        raise Exception("Predictors need to be callable")


class NodePrediction(_NodePredictor):
    def __call__(self, features):
        return tf.argmax(tf.gather(features, self.data), axis=1) # data is a list of nodes


class NodeLoss(_NodePredictor):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __call__(self, features):
        if self.transform is not None:
            features = self.transform(features)
        predictions = tf.nn.log_softmax(tf.gather(features, self.data["nodes"]), axis=1)
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(self.data["labels"], predictions)


class NodeAccuracy(_NodePredictor):
    def __init__(self, data):
        self.data = data

    def __call__(self, features):
        predictions = tf.argmax(tf.gather(features, self.data["nodes"]), axis=1)
        return 1-tf.math.count_nonzero(predictions-self.data["labels"]).numpy()/predictions.shape[0]


class GNN(Layered):
    def __init__(self, G: tf.Tensor, features: tf.Tensor, loss_transform=None):
        super().__init__(features.shape)
        self.G = G
        self.features = features
        self.loss_transform = loss_transform

    def predict(self, predictor: _NodePredictor):
        return predictor(self(self.features))

    def get_adjacency(self, graph_dropout=0.5, renormalized=True):
        G = self.sparse_dropout(self.G, graph_dropout)
        if renormalized:
            G = tf.sparse.add(G, tf.sparse.eye(G.shape[0]))
            D = 1. / tf.sqrt(tf.sparse.reduce_sum(G, axis=0))
            G = tf.reshape(D, (-1, 1)) * G * D
        return G

    def train(self, train_data, valid_data=None, test_data=None, patience=50, learning_rate=0.01, regularization=5.E-4, batch_size=None, verbose=True):
        self.reset()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if valid_data is None:
            valid_data = train_data
        min_loss = float('inf')
        min_loss_vars = [var.identity() for var in self.vars()]
        min_loss_output = None
        patience_remaining = patience
        for epoch in range(2000):
            self.training_mode(True)
            for train in batches(train_data, batch_size):
                vars = [var.var for var in self.vars() if var.trainable]
                with tf.GradientTape() as tape:
                    loss = self.predict(NodeLoss(train, self.loss_transform))
                    for var in self.vars():
                        loss += regularization * var.regularize * tf.nn.l2_loss(var.var)
                    gradients = tape.gradient(loss, vars)
                    optimizer.apply_gradients(zip(gradients, vars))
            output = self(self.features)
            self.training_mode(False)
            # learn loss and update previous loss
            if self.loss_transform is not None:
                self.loss_transform.training_mode(False)
                loss_transform_vars = [var.var for var in self.loss_transform.vars() if var.trainable]
                with tf.GradientTape() as tape:
                    train_loss = NodeLoss(train_data, self.loss_transform)(output)
                    valid_loss = NodeLoss(valid_data, self.loss_transform)(output)
                    loss = valid_loss + train_loss - tf.sqrt(tf.square(train_loss-valid_loss))
                    for var in self.loss_transform.vars():
                        loss += regularization * var.regularize * tf.nn.l2_loss(var.var)
                    gradients = tape.gradient(loss, loss_transform_vars)
                    optimizer.apply_gradients(zip(gradients, loss_transform_vars))
                self.loss_transform.training_mode(False)
                if min_loss_output is not None:
                    min_loss = float(NodeLoss(valid_data, self.loss_transform)(min_loss_output))

            # patience mechanism
            valid_loss = float(NodeLoss(valid_data, self.loss_transform)(output))
            patience_remaining -= 1
            if verbose and valid_loss < min_loss:
                test_acc = None if test_data is None else self.predict(NodeAccuracy(test_data))
                valid_acc = NodeAccuracy(valid_data)(output)
                print(f'Epoch {epoch}  patience {patience_remaining}  Valid. loss {valid_loss:.3f} vs {min_loss:.3f}  Valid. acc {valid_acc:.3f}  Test. acc {test_acc:.3f}')
            if valid_loss < min_loss:
                min_loss = valid_loss
                min_loss_vars = [var.identity() for var in self.vars()]
                min_loss_output = output
                patience_remaining = patience
            if patience_remaining == 0 and epoch > 10:
                break
        for var, best_var in zip(self.vars(), min_loss_vars):
            var.assign(best_var)