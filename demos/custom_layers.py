from experiments.experiment_setup import dgl_setup
import gnntf
import tensorflow as tf

G, labels, features, train, valid, test = dgl_setup("cora")
num_classes = len(set(labels))

gnn = gnntf.GNN(gnntf.graph2adj(G), features)
gnn.add(gnntf.Dense(64, activation=tf.nn.relu, dropout=0.6))
H0 = gnn.add(gnntf.Dense(num_classes, activation=tf.nn.relu, regularize=False))
for _ in range(10):
    gnn.add(gnntf.PPRIteration(H0, 0.1))

gnn.train(train=gnntf.NodeClassification(train, labels[train]),
          valid=gnntf.NodeClassification(valid, labels[valid]))


prediction = gnn.predict(gnntf.NodeClassification(test))
accuracy = gnntf.acc(prediction, labels[test])
print(accuracy)
