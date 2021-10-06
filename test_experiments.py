import numpy as np
import tensorflow as tf
from experiments.experiment_setup import dgl_setup
from core.gnn import NodePrediction
from core.gnn import APPNP, GCNII, APExp
from gnntf.measures import acc, set_seed

def experiments(create_gnn, dataset, repeats=20, **kwargs):
    G, labels, features, train, valid, test = dataset
    accs = list()
    for experiment in range(repeats):
        #train, valid, test = custom_splits(labels, num_validation=None, seed=experiment)
        set_seed(experiment)
        gnn = create_gnn()
        gnn.train({"nodes": train, "labels": labels[train]},
                         {"nodes": valid, "labels": labels[valid]},
                         {"nodes": test, "labels": labels[test]},
                         **kwargs)
        prediction = gnn.predict(NodePrediction(test))
        accuracy = acc(prediction, labels[test])
        accs.append(accuracy)
        print('Accuracy', np.mean(accs), '\pm', 1.96*np.std([np.random.choice(accs, len(accs), replace=True) for _ in range(1000)]))

datasets = ['cora']#, 'cora', 'pubmed']
with tf.device('/cpu:0'):
    for dataset_name in datasets:
        dataset = dgl_setup(dataset_name)
        G, labels, features, train, valid, test = dataset
        num_classes = len(set(labels))
        print('====== Dataset ======')
        print('Name', dataset_name)
        print('Features', features.shape[1])
        print('Edges', G.indices.shape[0])
        print('Classes', num_classes)
        loss_transform = None#Layered([G.shape[0], num_classes], [Activation("scale")])
        if dataset_name == "cora":
            gnn = lambda: GCNII(G, features, num_classes=num_classes, iterations=64, l=0.5, dropout=0.6, latent_dims=[256], convolution_regularization=20, loss_transform=loss_transform)
        if dataset_name == 'citeseer':
            gnn = lambda: GCNII(G, features, num_classes=num_classes, iterations=32, l=0.6, dropout=0.7, latent_dims=[256], convolution_regularization=20, loss_transform=loss_transform)
        elif dataset_name == 'pubmed':
            gnn = lambda: GCNII(G, features, num_classes=num_classes, iterations=16, l=0.4, dropout=0.5, latent_dims=[256], convolution_regularization=1, loss_transform=loss_transform)
        gnn = lambda: APPNP(G, features, num_classes=num_classes, loss_transform=loss_transform)
        gnn = lambda: APExp(G, features, num_classes=num_classes, loss_transform=loss_transform)
        experiments(gnn, dataset, 5, verbose=True, patience=100)
