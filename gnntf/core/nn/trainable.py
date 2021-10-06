import tensorflow as tf
from .layered import Layered


class Predictor(object):
    def predict(self, features: tf.Tensor):
        raise Exception("Predictors need to implement a predict method")

    def loss(self, features: tf.Tensor):
        raise Exception("Predictors need to implement a loss method")

    def evaluate(self, features: tf.Tensor):
        raise Exception("Predictors need to implement an evaluate method")


class Trainable(Layered):
    def __init__(self, features):
        super().__init__(features.shape)
        self.features = features

    def predict(self, predictor: Predictor):
        return predictor.predict(self(self.features))

    def train(self,
              train: Predictor,
              valid: Predictor = None,
              test: Predictor = None,
              patience: int = 100,
              learning_rate: float = 0.01,
              regularization: float = 5.E-4,
              verbose: bool = False):
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
            if verbose and valid_loss < min_loss:
                train_acc = float(train.evaluate(output))
                test_acc = float("nan") if test is None else float(test.evaluate(output))
                valid_acc = float(valid.evaluate(output))
                print(f'Epoch {epoch}  patience {patience_remaining}  Train loss {float(loss.numpy()):.3f} Validation loss {valid_loss:.3f}  Train {train_acc:.3f} Validation {valid_acc:.3f}  Test {test_acc:.3f}')
            if valid_loss < min_loss:
                min_loss, min_loss_vars = valid_loss, [var.identity() for var in self.vars()]
                patience_remaining = patience
            if patience_remaining == 0:
                break
        for var, best_var in zip(self.vars(), min_loss_vars):
            var.assign(best_var)
