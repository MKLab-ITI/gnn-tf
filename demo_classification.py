from experiments.experiment_setup import dgl_setup
from core.nn.architectures.mlp import MLP
from core.gnn.graph_predictor import NodeClassification
from utils import acc, set_seed

set_seed(0)
G, labels, features, train, valid, test = dgl_setup("cora")
num_classes = len(set(labels))
gnn = MLP(features, num_classes=num_classes)
#gnn = APPNP(G, features, num_classes=num_classes, enable_error=False)

gnn.train(train=NodeClassification(train, labels[train]),
          valid=NodeClassification(valid, labels[valid]),
          test=NodeClassification(test, labels[test]),
          patience=100)

prediction = gnn.predict(NodeClassification(test))
accuracy = acc(prediction, labels[test])
print(accuracy)