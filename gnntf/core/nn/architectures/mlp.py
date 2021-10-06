from gnntf.core.nn.trainable import Trainable
from gnntf.core.nn.layers import Dropout, Dense
import tensorflow as tf


class MLP(Trainable):
    def __init__(self, features: tf.Tensor, num_classes: int, latent_dims=[64], dropout: float = 0.5):
        super().__init__(features)
        self.add(Dropout(dropout))
        for latent_dim in latent_dims:
            self.add(Dense(latent_dim, dropout=dropout, activation=tf.nn.relu))
        self.add(Dense(num_classes, dropout=0, regularize=False))