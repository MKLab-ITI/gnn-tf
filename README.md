# TestGNN: A Flexible Deep GNN Framework
This repository provides a GNN framework for easy experimentation with deep GNN architectures.

Here is an example of defining the popular APPNP achitecture, which combines
Personalized PageRank with Graph Convolutional Networks.

```python
from gcn import ClassificationGCN
import tensorflow as tf

class APPNP(ClassificationGCN):
    def __init__(self, *args, **kwargs):
        super(APPNP, self).__init__(*args, **kwargs)
        self.a = self.create_var(shape=(1, 1))

    def build_layer(self, layer_num, input_dims, output_dims):
        pass

    def preprocess(self, features):
        features = super().preprocess(features)
        self.H0 = features
        return features

    def call_layer(self, _, features):
        features = (1-self.a)*tf.sparse.sparse_dense_matmul(self.adjacency_matrix, features) + self.a*self.H0
        return features
```