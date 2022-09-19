# Predictive Tasks
There are two predictive tasks currently supported: 
node classification and link prediction. Predictive tasks are
decoupled from the GNN architecture they model and are passed
to the GNN's ``train`` method to define the training and validation
objectives.

1. [Node Classification](#node-classification)
2. [Link Prediction](#link-prediction)

# Node Classification
The following code
demonstrates an example of how to use pass a ``NodeClassification``
predictive task to the GNN to let it know what to train towards.

```python
from experiments.experiment_setup import dgl_setup
import gnntf

gnntf.set_seed(0)
G, labels, features, train, valid, test = dgl_setup("cora")
num_classes = len(set(labels))
gnn = gnntf.APPNP(G, features, num_classes=num_classes)

gnn.train(train=gnntf.NodeClassification(train, labels[train]),
          valid=gnntf.NodeClassification(valid, labels[valid]))

prediction = gnn.predict(gnntf.NodeClassification(test))
accuracy = gnntf.acc(prediction, labels[test])
print(accuracy)
```


# Link Prediction
```python
from experiments.experiment_setup import dgl_setup
import gnntf
import random

gnntf.set_seed(0)
G, _, features = dgl_setup("cora")[:3]
adj = gnntf.graph2adj(G)
edges = adj.indices.numpy()
train = random.sample(range(len(edges)), int(len(edges) * 0.8))
valid = random.sample(list(set(range(len(edges))) - set(train)), (len(edges)-len(train))//4)
test = list(set(range(len(edges))) - set(valid) - set(train))

training_graph = gnntf.create_nx_graph(G, edges[train])

gnn = gnntf.APPNP(gnntf.graph2adj(training_graph), features, num_classes=16, positional_dims=16)
gnn.train(train=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[train], G)),
          valid=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[valid], G)),
          test=gnntf.LinkPrediction(*gnntf.negative_sampling(edges[test], G)),
          patience=50, verbose=True)

edges, labels = gnntf.negative_sampling(edges[test], G)
prediction = gnn.predict(gnntf.LinkPrediction(edges))
print(gnntf.auc(labels, prediction))
```

```python
import numpy as np
from experiments.experiment_setup import dgl_setup
G = dgl_setup("cora")[0]
features = np.zeros((len(G),1))
```