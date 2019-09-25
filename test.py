from data.nutella import load
import networkx as nx
import numpy as np
import gnn



G = load()
adjacency = np.array(nx.to_numpy_matrix(G)).astype('float32')
features = np.identity(len(G)).astype('float32')
convolution = adjacency/np.sum(adjacency, axis=1)

#training_edges = np.array([[i, j] for i in range(len(G)) for j in range(len(G))])
training_edges = gnn.negative_sampling(adjacency)
training_labels = np.expand_dims(np.array([adjacency[training_edges[i][0],training_edges[i][1]] for i in range(len(training_edges))]), axis=1).astype('float32')

model = gnn.Model(len(G), [32])
model.train(convolution, features, training_edges, training_labels, epochs=50)

print('AUC', model.evaluate(convolution, features, training_edges, training_labels))
