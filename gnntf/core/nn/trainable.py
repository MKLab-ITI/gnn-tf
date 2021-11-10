import tensorflow as tf
from .layered import Layered
from tqdm import tqdm

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
        self._fast_predict = None

    def reset(self):
        super().reset()
        self._fast_predict = None

    def predict(self, predictor: Predictor):
        if self._fast_predict is None:
            self._fast_predict = self(self.features)
        return predictor.predict(self._fast_predict)

    def loss(self, predictor: Predictor):
        if self._fast_predict is None:
            self._fast_predict = self(self.features)
        return predictor.loss(self._fast_predict)

    def evaluate(self, predictor: Predictor):
        if self._fast_predict is None:
            self._fast_predict = self(self.features)
        return predictor.evaluate(self._fast_predict)

    def train(self,
              train: Predictor,
              valid: Predictor = None,
              test: Predictor = None,
              patience: int = 100,
              learning_rate: float = 0.01,
              regularization: float = 5.E-4,
              verbose: bool = False,
              epochs: int = 2000,
              degradation = lambda epoch: 1,
              batches: int = 1):
        self.reset()
        #if isinstance(learning_rate, float):
        #    learning_rate = lambda _: learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        if valid is None:
            valid = train
        min_loss = float('inf')
        min_loss_vars = [var.identity() for var in self.vars()]
        patience_remaining = patience
        for epoch in range(epochs):
            if self._fast_predict is not None:
                del self._fast_predict
            #optimizer._set_hyper("learning_rate", learning_rate(epoch))
            loss = 0
            for _ in range(batches):
                with self as vars:
                    with tf.GradientTape() as tape:
                        batch_loss = train.loss(self(self.features))
                        for layer in self.layers():
                            if layer.output_regularize != 0:
                                batch_loss = batch_loss + layer.loss() * regularization
                        for var in self.vars():
                            if var.regularize != 0:
                                batch_loss += regularization * var.regularize * tf.nn.l2_loss(var.var)
                        gradients = tape.gradient(batch_loss* degradation(epoch), vars)
                    optimizer.apply_gradients(zip(gradients, vars))
                    loss = loss + float(batch_loss.numpy())

            # patience mechanism
            output = self(self.features)
            valid_loss = float(valid.loss(output))
            patience_remaining -= 1
            if verbose:# and valid_loss < min_loss:
                train_acc = float(train.evaluate(output))
                test_acc = float("nan") if test is None else float(test.evaluate(output))
                valid_acc = float(valid.evaluate(output))
                print(f'Epoch {epoch}  patience {patience_remaining}  Train loss {float(loss):.3f} Validation loss {valid_loss:.3f}  Train {train_acc:.3f} Validation {valid_acc:.3f}  Test {test_acc:.3f}')
            if valid_loss < min_loss:
                min_loss, min_loss_vars = valid_loss, [var.identity() for var in self.vars()]
                patience_remaining = patience
            if patience_remaining == 0:
                break
        for var, best_var in zip(self.vars(), min_loss_vars):
            var.assign(best_var)
