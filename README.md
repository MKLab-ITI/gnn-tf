# TestGNN: A Flexible Deep Graph Neural Network Framework
This repository provides a framework for easy experimentation 
with Graph Neural Network (GNN) architectures. 

The following documentation is still under construction.

## GNN Architectures
A GNN architecture can be either imported from one of the implemented 
state-of-the-art ones or can be customly defined.

### Implemented GNNs
Currently implemented architectures.

Architecture | Reference 
| ----------- | ----------- |
``from core.gnn import APPNP`` | [TODO]
``from core.gnn import GCNII`` | [TODO]


### Custom GNNs
Custom GNNs can be defined by extended the GNN class and adding layers
during the constructor method. Typical Neural Network layers can be
found in the module ``core.gnn.nn.layers``. For example, a traditional
perceptron with two dense layers and dropout to be used for classification
can be defined per the following code. 

```python
from core.nn.layers import Dropout, Dense
from core.gnn.gnn import GNN
import tensorflow as tf

class CustomGNN(GNN):
    def __init__(self, G: tf.Tensor, features: tf.Tensor, hidden_layer=64, num_classes=3, **kwargs):
        super().__init__(G, features, **kwargs)
        self.add(Dropout(0.5))
        self.add(Dense(hidden_layer, dropout=0.5, activation=tf.nn.relu))
        self.add(Dense(num_classes, dropout=0, regularize=False))
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
from core.gnn import APPNP, NodeClassification
from utils import acc, set_seed

set_seed(0)
G, labels, features, train, valid, test = dgl_setup("cora")
num_classes = len(set(labels))
gnn = APPNP(G, features, num_classes=num_classes)

gnn.train(train=NodeClassification(train, labels[train]),
          valid=NodeClassification(valid, labels[valid])  )

prediction = gnn.predict(NodeClassification(test))
accuracy = acc(prediction, labels[test])
print(accuracy)
```


### Link Prediction
[TODO]