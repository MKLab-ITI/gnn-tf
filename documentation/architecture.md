# Architectures
GNN architectures can be imported from a list of implemented
ones, but new ones can also be defined by instantiating.

1. [Literature Implementations](#literature-implementations)
2. [Defining New Architectures](#defining-new-architectures)
3. [Quick Customization](#quick-customization)
4. [Wrapping Keras Layers](#wrapping-keras-layers)

# Literature Implementations
The following architectures are currently implemented: 
GCN, GCNII, APPNP. You can import them from the package's
top level. For example, instantiating the APPNP architecture
is as simple as calling:

```python
import gnntf
G, features, num_classes = ...
gnn = gnntf.APPNP(gnntf.graph2adj(G), features, num_classes=num_classes)
```

Contrary to general-purpose machine learning frameworks, you need to 
provide the graph (e.g. starting from a networkx graph `G`) and
node features during architecture definition. This means that the
architecture is defined for specific input data.


# Defining New Architectures
Custom GNNs can be defined by extending the GNN class and adding layers
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

In addition to typical functionalities provided by neural network libraries,
we also provide flow control functionality on the layer level that removes the need
of understanding tensorflow primitives at all by using `Branch` and `Concatenate` 
layers.

# Quick Customization
You can simplify architecture definition by instantiating it
via `gnntf.GNN(...)` and adding layers afterwards.
In fact, you can add layers after training just fine or
even remove top layers by calling

# Wrapping Keras Layers
You can turn Keras layers by wrapping them with the gnntf
interface by calling `layer = gnntf.Wrap(layer)`. This
can be added to architectures.