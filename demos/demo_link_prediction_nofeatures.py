from experiments.experiment_setup import tpl_setup
import gnntf
import random
import numpy as np
import tensorflow as tf

gnntf.set_seed(0)
graph = tpl_setup()[0]
features = np.zeros((len(graph), 0))
adj = gnntf.graph2adj(graph)
edges = adj.indices.numpy()
train = random.sample(range(len(edges)), int(len(edges) * 0.8))
valid = random.sample(list(set(range(len(edges))) - set(train)), (len(edges)-len(train))//4)
test = list(set(range(len(edges))) - set(valid) - set(train))
training_graph = gnntf.create_nx_graph(list(range(len(graph))), edges[train])

gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=8, positional_dims=8)
gnn.train(train=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[train], training_graph, samples=1)),
          valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph, samples=1)),
          test=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[test], graph, samples=1)),
          patience=50, verbose=True)

edges, labels = gnntf.negative_sampling(edges[test], graph)
prediction = gnn.predict(gnntf.LinkPrediction(edges))
print(gnntf.auc(labels, prediction))
