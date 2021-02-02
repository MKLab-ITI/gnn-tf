# TestGNN: A Flexible Deep GNN Framework
This repository provides a GNN framework for easy experimentation with deep GNN architectures.

Here is an example of defining the popular APPNP achitecture, which combines
Personalized PageRank with Graph Convolutional Networks.

```python
from core.gcn import ClassificationGCN
from core import layers

class APPNP(ClassificationGCN):
    def __prebuild__(self, num_classes, restart_probability=0.1, latent_dims=[64], iterations=10):
        for latent_dim in latent_dims:
            self.add(layers.Dense(latent_dim))
        self.add(layers.Dense(num_classes))
        H0 = self.add(layers.ResidualValue())
        for _ in range(iterations):
            self.add(layers.PPRIteration(H0, restart_probability))
```