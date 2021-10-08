from experiments.experiment_setup import tpl_setup
import gnntf
import random
import numpy as np
import tensorflow as tf

gnntf.set_seed(0)
graph = tpl_setup()[0]
print("Nodes", len(graph))
print("Dependencies", graph.number_of_edges())

node2id = {node: i for i, node in enumerate(graph)}
features = np.array([[0. if "A" in node else 1.] for node in graph]) # labeling trick per https://arxiv.org/pdf/2010.16103.pdf
aucs = list()
precs = list()
recs = list()
f1s = list()
maps = list()
edges = gnntf.graph2adj(graph, directed=False).indices.numpy()
rm = 3
k = 5

libraries = [node2id[node] for node in graph if "L" in node]

for node in graph:
    if graph.degree(node) >= 10 and "A" in node:
        node = node2id[node]
        node_edges = [i for i in range(len(edges)) if edges[i][0] == node or edges[i][1] == node]
        test = random.sample(node_edges, rm)
        non_test = list(set(range(len(edges)))-set(test))
        valid = random.sample(non_test, len(non_test)//4)
        train = list(set(range(len(edges)))-set(valid)-set(test))
        training_graph = gnntf.create_nx_graph(list(range(len(graph))), edges[train])

        gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=16, positional_dims=16)
        gnn.train(train=gnntf.LinkPrediction(lambda: gnntf.negative_sampling(edges[random.sample(train,128)],
                                                                             training_graph, samples=1, negative_nodes=libraries),
                                             ),
                  valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], training_graph, samples=1, negative_nodes=libraries)),
                  patience=100, verbose=True)

        test_links, labels = gnntf.negative_sampling(edges[test], graph, samples=100, negative_nodes=libraries)
        prediction = gnn.predict(gnntf.LinkPrediction(test_links)).numpy()
        aucs.append(gnntf.auc(labels, prediction))
        maps.append(gnntf.avprec(labels, prediction, k))
        precs.append(gnntf.prec(labels, prediction, k))
        recs.append(gnntf.rec(labels, prediction, k))
        f1s.append(gnntf.f1(labels, prediction, k))
        print(f"Average node AUC {float(np.mean(aucs)):.3f}\t "
              f"MAP {float(np.mean(maps)):.3f}\t"
              f"Precision {float(np.mean(precs)):.3f}\t"
              f"Recall {float(np.mean(recs)):.3f}\t "
              f"F1 {float(np.mean(f1s)):.3f}")