import tensorflow as tf
from core.nn.layered import Layered
from core.gnn.predictor import Predictor


class GNN(Layered):
    def __init__(self, G: tf.Tensor, features: tf.Tensor):
        super().__init__(features.shape)
        self.G = G
        self.features = features

    def predict(self, predictor: Predictor):
        return predictor.predict(self(self.features))

    def get_adjacency(self, graph_dropout=0.5, renormalized=True):
        G = self.sparse_dropout(self.G, graph_dropout)
        if renormalized:
            G = tf.sparse.add(G, tf.sparse.eye(G.shape[0]))
            D = 1. / tf.sqrt(tf.sparse.reduce_sum(G, axis=0))
            G = tf.reshape(D, (-1, 1)) * G * D
        return G

    def train(self, train: Predictor, valid: Predictor = None, test: Predictor = None,
              patience=50, learning_rate=0.01, regularization=5.E-4):
        self.reset()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if valid is None:
            valid = train
        min_loss = float('inf')
        min_loss_vars = [var.identity() for var in self.vars()]
        patience_remaining = patience
        for epoch in range(2000):
            with self as vars:
                with tf.GradientTape() as tape:
                    loss = train.loss(self(self.features))
                    for var in self.vars():
                        loss += regularization * var.regularize * tf.nn.l2_loss(var.var)
                    gradients = tape.gradient(loss, vars)
                    optimizer.apply_gradients(zip(gradients, vars))

            # patience mechanism
            output = self(self.features)
            valid_loss = float(valid.loss(output))
            patience_remaining -= 1
            if test is not None and valid_loss < min_loss:
                test_acc = float(test.evaluate(output))
                valid_acc = float(valid.evaluate(output))
                print(f'Epoch {epoch}  patience {patience_remaining}  Validation loss {valid_loss:.3f}  Validation {valid_acc:.3f}  Test {test_acc:.3f}')
            if valid_loss < min_loss:
                min_loss, min_loss_vars = valid_loss, [var.identity() for var in self.vars()]
                patience_remaining = patience
            if patience_remaining == 0:
                break
        for var, best_var in zip(self.vars(), min_loss_vars):
            var.assign(best_var)