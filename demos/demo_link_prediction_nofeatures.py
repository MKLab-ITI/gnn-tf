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

gnn = gnntf.GRec(gnntf.graph2adj(training_graph), features, num_classes=128, positional_dims=128)
gnn.train(train=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[train], training_graph, samples=1)),
          valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph, samples=1)),
          test=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[test], graph, samples=1)),
          patience=5, verbose=True)

edges, labels = gnntf.negative_sampling(edges[test], graph)
prediction = gnn.predict(gnntf.LinkPrediction(edges))
print("AUC over all edges", gnntf.auc(labels, prediction))

node2id = {v: i for i, v in enumerate(graph)}
aucs = list()
precs = list()
recs = list()
f1s = list()
for node in graph:
    if graph.degree(node) >= 10: #and "A" in node:
        edges = np.array([(node2id[node], node2id[neighbor]) for neighbor in graph])
        labels = [1 if graph.has_edge(node, neighbor) or graph.has_edge(neighbor, node) else 0 for neighbor in graph]
        if max(labels) != min(labels):
            prediction = gnn.predict(gnntf.LinkPrediction(edges)).numpy()
            aucs.append(gnntf.auc(labels, prediction))
            precs.append(gnntf.prec(labels, prediction))
            recs.append(gnntf.rec(labels, prediction))
            f1s.append(gnntf.f1(labels, prediction))
print(f"Average node AUC {float(np.mean(aucs)):.3f}\t "
      f"Precision {float(np.mean(precs)):.3f}\t"
      f"Recall {float(np.mean(recs)):.3f}\t "
      f"F1 {float(np.mean(f1s)):.3f}")