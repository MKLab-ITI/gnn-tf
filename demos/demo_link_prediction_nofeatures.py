from experiments.experiment_setup import dgl_setup
import gnntf
import random
import numpy as np

gnntf.set_seed(0)
graph = dgl_setup("cora")[0]
features = np.zeros((len(graph), 0))
edges = gnntf.graph2adj(graph, directed=False).indices.numpy()
train = random.sample(range(len(edges)), int(len(edges) * 0.8))
valid = random.sample(list(set(range(len(edges))) - set(train)), (len(edges)-len(train))//4)
test = list(set(range(len(edges))) - set(valid) - set(train))
training_graph = gnntf.create_nx_graph(list(range(len(graph))), edges[train])

gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=128, positional_dims=128)
gnn.train(train=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[train], training_graph, samples=1)),
          valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph, samples=1)),
          test=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[test], graph, samples=1)),
          patience=100, verbose=True)

edges, labels = gnntf.negative_sampling(edges[test], graph)
prediction = gnn.predict(gnntf.LinkPrediction(edges))
print("AUC over all edges", gnntf.auc(labels, prediction))
