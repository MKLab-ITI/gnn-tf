import tensorflow as tf
import numpy as np

def acc(labels, predictions):
    return sum([1. for label, prediction in zip(labels, predictions) if (label>0.5) == (prediction>0.5)])/len(predictions)


def auc(labels, predictions):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)


class GCNModel(object):
    def __init__(self, adjacency_matrix, node_features, embedding_dims, *args, **kwargs):
        self.vars = list()
        embedding_dims = embedding_dims
        if node_features is None:
            self.features = self.create_var((adjacency_matrix.shape[0], embedding_dims[0]), 'positive uniform') # node features
            self.feature_embeddings = None
        else:
            self.features = tf.Variable(node_features, shape=node_features.shape, dtype='float32')
            self.feature_embeddings = self.create_var(shape=(node_features.shape[1], embedding_dims[0]))
        self.layers = list()
        for i in range(1,len(embedding_dims)):
            self.layers.append(self.build_layer(i-1, embedding_dims[i-1], embedding_dims[i]))
        self.build_predictor(embedding_dims[-1])
        self.adjacency_matrix = adjacency_matrix

    def create_var(self, shape, normalization='xavier', learnable=True):
        multiply = 0
        if normalization == 'he':
            multiply = (0.5/shape[1])**0.5
        elif normalization == 'xavier':
            multiply = (1/(shape[0]+shape[1]))**0.5
        elif normalization == 'uniform':
            multiply = 1
        elif normalization == 'positive uniform':
            var = tf.Variable(tf.random.uniform(shape=shape)/shape[1])
            multiply = None
        if multiply is not None:
            multiply *= 6**0.5
            var = tf.Variable(multiply*tf.random.uniform(shape=shape))
        if learnable:
            self.vars.append(var)
        return var

    def build_layer(self, layer_num, input_dims, output_dims):
        raise Exception("Should implement the build_layer method")

    def call_layer(self, layer, features):
        raise Exception("Should implement the call layer method")

    def build_predictor(self, input_dims):
        raise Exception("Should implement the build predictor method")

    def call_predictor(self, features, data):
        raise Exception("Should implement the call predictor method")

    def loss(self, labels, predictions):
        raise Exception("Should implement the loss method")

    def predict(self, data):
        features = self.features
        if self.feature_embeddings is not None:
            features = tf.matmul(features, self.feature_embeddings)
        for layer in self.layers:
            features = self.call_layer(layer, features)
        return self.call_predictor(features, data)

    def _train_batch(self, data, labels, optimizer, regularization_weight, regularize_intermediates=True):
        labels = tf.convert_to_tensor(labels)
        with tf.GradientTape() as tape:
            if regularize_intermediates and len(self.layers) > 1:
                loss = 0
                features = self.features
                if self.feature_embeddings is not None:
                    features = tf.matmul(features, self.feature_embeddings)
                for layer in self.layers:
                    features = self.call_layer(layer, features)
                    predictions = self.call_predictor(features, data)
                    loss += self.loss(labels, predictions)
            else:
                predictions = self.predict(data)
                loss = self.loss(labels, predictions)
            regularization = 0
            for var in self.vars:
                regularization += tf.nn.l2_loss(var)
            loss = loss + regularization_weight * regularization
        gradients = tape.gradient(loss, self.vars)
        optimizer.apply_gradients(zip(gradients, self.vars))
        return loss, predictions.numpy().flatten().tolist()

    def train(self, data, labels, batch_size=None, validation_data=None, validation_labels=None,
              epochs=100, learning_rate=0.01, regularization_weight=5.E-4, early_stopping=10):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if batch_size is None:
            batch_size = len(labels)
        if validation_data is None:
            validation_data = data
        if validation_labels is None:
            validation_labels = labels
        prev_loss = float('inf')
        early_stopping_countdown = early_stopping
        for epoch in range(epochs):
            batch_start = 0
            loss = 0
            predictions = list()
            while batch_start < len(labels):
                batch_end = min(batch_start+batch_size, len(labels))
                #if len(data.shape)>1:
                   #batch_data = data[batch_start:batch_end,:]
                #else:
                batch_data = data[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]
                batch_loss, batch_predictions = self._train_batch(batch_data, batch_labels, optimizer, regularization_weight)
                loss += batch_loss
                predictions += batch_predictions
                batch_start += batch_size
            loss = float(loss)/len(labels)
            if loss < prev_loss:
                prev_loss = loss
                early_stopping_countdown = early_stopping
            else:
                early_stopping_countdown -= 1
            if early_stopping_countdown < 0:
                break
            #print('Epoch', epoch, '/', epochs, '\tLoss', loss, '\tacc', acc(labels, predictions), '\tAUC', auc(labels, predictions))
            #validation_predictions = self.predict(validation_data).numpy().flatten().tolist()
            validation_predictions = self.predict(validation_data).numpy()
            #print('Epoch', epoch, '/', epochs, '\tLoss', loss, '\tacc', acc(validation_labels, validation_predictions), '\tAUC', auc(validation_labels, validation_predictions))
            validation_predictions = np.argmax(validation_predictions,axis=1).tolist()
            print('Epoch', epoch, '/', epochs, '\tLoss', loss, '\tacc', acc(validation_labels, validation_predictions))
        return acc(validation_labels, validation_predictions)

class RelationalGCN(GCNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_predictor(self, input_dims):
        self.r = self.create_var(shape=(input_dims, 1))

    def call_predictor(self, features, data):
        # data are edges
        logits = tf.matmul(tf.multiply(tf.gather(features, data[:,0]), tf.gather(features, data[:,1])), self.r)
        return tf.nn.sigmoid(logits)

    def loss(self, labels, predictions):
        return tf.keras.losses.BinaryCrossentropy()(labels, predictions)


class ClassificationGCN(GCNModel):
    def __init__(self, *args, output_labels, **kwargs):
        self.output_labels = output_labels
        super().__init__(*args, **kwargs)

    def build_predictor(self, input_dims):
        self.Wout = self.create_var(shape=(input_dims, self.output_labels), normalization='he')

    def call_predictor(self, features, data):
        logits = tf.matmul(tf.gather(features, data), self.Wout)
        return tf.nn.softmax(logits)

    def loss(self, labels, predictions):
        return tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)