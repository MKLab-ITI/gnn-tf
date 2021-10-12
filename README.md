# gnntf: A Flexible Deep Graph Neural Network Framework
This repository provides a framework for easy experimentation 
with Graph Neural Network (GNN) architectures by separating
them from predictive components.

The following documentation is still under construction.


# Quickstart
**Data.** Let us first import a node classification dataset using
a helper method. This creates a networkx graph, an array
of integer node labels, a matrix of node features and
three arrays  indicating training, validation and test nodes.
The node  order is considered as the order of nodes in the graph
(i.e. the order accessed by the iterator `node for node in graph`).

```python
from experiments.experiment_setup import dgl_setup
graph, labels, features, train, valid, test = dgl_setup("cora")
```

**Architecture.** Let us define an `APPNP` GNN architecture that outputs a number of
representations equal to the number of classes. This requires as
input the graph in adjacency matrix form `adj`, the number of 
classes `num_classes` and known node features. Note that the 
adjacency matrix is constructed to be symmetric by filling
missing opposite direction edges.
```python
import gnntf
num_classes = len(set(labels))
adj = gnntf.graph2adj(graph)
architecture = gnntf.APPNP(adj, features, num_classes=num_classes)
```

**Predictive tasks.** Then we aim to train this architecture for node classification.
In this task, a softmax is applied on output representations
and the model is trained towards minimizing a cross-entropy loss.
We need to first define the training and validation tasks - these
could be different from each other and even different from the
prediction task.

```python
train_task = gnntf.NodeClassification(train, labels[train])
validation_task = gnntf.NodeClassification(valid, labels[valid])
```

**Training.** Then the architecture can be trained on these tasks.
Training can be customized with respect to the employed optimizer and
early stopping patience (training stops when validation assessment
does not improve for that number of epochs). In this example, we limit
running time with a small patience of 10 epochs, though the default
value for most GNN approaches is 100.

```python
architecture.train(train=train_task, valid=validation_task, patience=10)
```

**Prediction.** Finally, we can use the architecture to make predictions about a 
new test task - this could also differ from training and validation
ones. The format of predictions is determined by the type of task,
but is usually a numpy array of outputs correspondning to inputs.
Note that, for testing, we ommitred the known label
argument, since it is not needed for predictions. In the following code
we use a helper method to compute the accuracy of the predictions
made in our dataset.

```python
test_task = gnntf.NodeClassification(test)
prediction = architecture.predict(test_task)
accuracy = gnntf.acc(prediction, labels[test])
print(accuracy)
```


# GNN Architectures
GNN architectures can be imported from a list of implemented
ones, but new ones can also be defined.

### Implemented Architectures
The following architectures are currently implemented.

Architecture | Reference 
| ----------- | ----------- |
``from gnntf import APPNP`` | [TODO]
``from gnntf import GCNII`` | [TODO]


### Custom Architectures
Custom GNNs can be defined by extended the GNN class and adding layers
during the constructor method. Typical Neural Network layers can be
found in the module ``core.gnn.nn.layers``. For example, a traditional
perceptron with two dense layers and dropout to be used for classification
can be defined per the following code.

```python
import gnntf
import tensorflow as tf

class CustomGNN(gnntf.GNN):
    def __init__(self, 
                 G: tf.Tensor,
                 features: tf.Tensor, 
                 hidden_layer: int = 64, 
                 num_classes: int = 3, 
                 **kwargs):
        super().__init__(G, features, **kwargs)
        self.add(gnntf.Dropout(0.5))
        self.add(gnntf.Dense(hidden_layer, activation=tf.nn.relu))
        self.add(gnntf.Dropout(0.5))
        self.add(gnntf.Dense(num_classes,  regularize=False))
```

:warning: The dropout argument is applied for the time being on layer *outputs*.

:bulb: In addition to typical functionalities provided by neural network libraries,
we also provide flow control functionality on the layer level that removes the need
of understanding tensorflow primitives at all by using Branch and Concatenate layers.



### Custom Layers
[TODO]


## Predictive Tasks
There are two predictive tasks currently supported: 
node classification and link prediction. Predictive tasks are
decoupled from the GNN architecture they model and are passed
to the GNN's ``train`` method to define the training and validation
objectives.

### Experiment Setups
[TODO]

### Node Classification
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


### Link Prediction
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