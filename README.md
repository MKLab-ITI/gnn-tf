# gnntf: A Flexible Deep Graph Neural Network Framework
This repository provides a framework for easy experimentation 
with Graph Neural Network (GNN) architectures by separating
them from predictive components.

# :zap: Quickstart
**Data.** Let us first import a node classification dataset using
a helper method. This creates a `networkx` graph, an array
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
but is usually a numpy array of outputs corresponding to inputs.
Note that, for testing, we ommitted the known label
argument, since it is not needed for predictions. In the following code
we use a helper method to compute the accuracy of the predictions
made in our dataset.

```python
test_task = gnntf.NodeClassification(test)
prediction = architecture.predict(test_task)
accuracy = gnntf.acc(prediction, labels[test])
print(accuracy)
```

# :rocket: Features
* GNNs
* Rapid prototyping
* Modular architectures
* Predictive tasks independent of architectures
* Common training and losses
* Use tensorflow code (e.g. activations, optimizers, keras layers)
* Easy layer injection/removal

# :link: Material
[Architectures](documentation/architecture.md)<br>
[Predictive Tasks](documentation/tasks.md)