from experiments.experiment_setup import dgl_setup
import gnntf
import random

gnntf.set_seed(0)
graph, _, features = dgl_setup("cora")[:3]
adj = gnntf.graph2adj(graph)
edges = adj.indices.numpy()
train = random.sample(range(len(edges)), int(len(edges) * 0.8))
valid = random.sample(list(set(range(len(edges))) - set(train)), (len(edges)-len(train))//4)
test = list(set(range(len(edges))) - set(valid) - set(train))
training_graph = gnntf.create_nx_graph(graph, edges[train])

gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=16)
gnn.train(train=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[train], training_graph)),
          valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph)),
          patience=100)

edges, labels = gnntf.negative_sampling(edges[test], graph)
prediction = gnn.predict(gnntf.LinkPrediction(edges))
print(gnntf.auc(labels, prediction))
