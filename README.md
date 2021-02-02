# TestGNN: A Flexible Deep GNN Framework
This repository provides a GNN framework for easy experimentation with deep GNN architectures.

Documentation is under contruction. This hands-on example of how to use this library:

```python
from core.gnn.filter import APPNP as Architecture
from core.gnn.gnn import NodePrediction
from experiments.experiment_setup import dgl_setup
from utils import acc, set_seed

set_seed(0) # ensures reproducibility

G, labels, features, train, valid, test = dgl_setup("cora") # can also try citeseer and pubmed
num_classes = len(set(labels))

gnn = Architecture(G, features, num_classes=num_classes)
gnn.train(  train={"nodes": train, "labels": labels[train]}, # data for train node ids
            valid={"nodes": valid, "labels": labels[valid]}, # data for valid(ation) node ids
            verbose=False, patience=100) # patience is the number of epochs without loss decrease before stopping
prediction = gnn.predict(NodePrediction(test)) # NodePrediction sets up a prediction task based on test nodes
print('Accuracy', acc(prediction, labels[test]))
```