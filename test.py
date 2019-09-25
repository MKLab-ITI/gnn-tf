from data.nutella import load
import networkx as nx
import numpy as np
import gnn

def get_labels(adjacency, edges):
    return np.expand_dims(np.array([adjacency[edges[i][0], edges[i][1]] for i in range(len(edges))]), axis=1).astype('float32')

G = load()
adjacency = np.array(nx.to_numpy_matrix(G)).astype('float32')
features = np.identity(len(G)).astype('float32')
convolution = adjacency/np.sum(adjacency, axis=1)

#training_edges = np.array([[i, j] for i in range(len(G)) for j in range(len(G))])
node_ids = list(range(len(G)))
training_edges = gnn.negative_sampling(adjacency, node_ids[100:])
test_edges = gnn.negative_sampling(adjacency, node_ids[:100])


model = gnn.Model(len(G), [32])
model.train(convolution, features, training_edges, get_labels(adjacency, training_edges), epochs=50)
print('Training AUC', model.evaluate(convolution, features, training_edges, get_labels(adjacency, training_edges)))
print('Validation AUC', model.evaluate(convolution, features, test_edges, get_labels(adjacency, test_edges)))

