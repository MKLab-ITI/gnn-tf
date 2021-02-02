import numpy as np
import random
import tensorflow as tf
from experiments.experiment_setup import dgl_setup, custom_splits
from core.gnn.gnn import NodePrediction
from core.gnn.filter import APPNP as Architecture
from core.gnn.gcn import GCNII as Architecture
from core.gnn.filter import PPRIteration
from core.nn.layers import Activation, Layered, Dense
from utils import acc, set_seed


dataset = "cora"

G, labels, features, train, valid, test = dgl_setup(dataset)
num_classes = len(set(labels))

print('====== Data characteristics ======')
print('Features', features.shape[1])
print('Edges', G.indices.shape[0])
print('Label types', num_classes)

np.random.seed(0)
random.shuffle(valid)
#valid = valid[:len(train)]

with tf.device('/cpu:0'):
    accs = list()
    for experiment in range(20):
        #train, valid, test = custom_splits(labels, num_validation=None, seed=experiment)
        set_seed(experiment)

        classifier = Architecture(G, features, num_classes=num_classes)#, loss_transform=Layered([G.shape[0], num_classes], [Activation("scale")]))
        classifier.train({"nodes": train, "labels": labels[train]},
                         {"nodes": valid, "labels": labels[valid]},
                         {"nodes": test, "labels": labels[test]},
                         verbose=True, patience=100)
        prediction = classifier.predict(NodePrediction(test))
        accuracy = acc(prediction, labels[test])
        accs.append(accuracy)
        print('Accuracy', np.mean(accs), '\pm', 1.96*np.std([np.random.choice(accs, len(accs), replace=True) for _ in range(1000)]))
