from data.pairs import facebook as load
import networkx as nx
import numpy as np
import gnn
import numpy.random

def get_labels(adjacency, edges):
    return np.expand_dims(np.array([adjacency[i, j] for i,j in edges]), axis=1).astype('float32')


print("Loading...")
G = load()
print(len(G), "nodes,", G.number_of_edges(), "edges")
adjacency = np.array(nx.to_numpy_matrix(G)).astype('float32')
features = np.identity(len(G)).astype('float32')

print("Creating training/test data...")
edges = gnn.negative_sampling(adjacency)
training_ids = numpy.random.choice(list(range(len(G))), len(G)*8//10)
training_edges = edges[training_ids,:]
test_edges = edges[np.setdiff1d(edges, training_ids),:]

convolution = np.copy(adjacency)
#for i,j in training_edges:
    #convolution[i,j] = 0
convolution = convolution/np.sum(convolution, axis=1)
training_labels = get_labels(adjacency, training_edges)
test_labels = get_labels(adjacency, test_edges)
print("Creating model...")
model = gnn.Model(len(G), [16, 16])

print("Training...")
model.train(convolution, features, training_edges, training_labels, epochs=50)
print('Testing...')
print('Training AUC', model.evaluate(convolution, features, training_edges, training_labels))
print('Test AUC', model.evaluate(convolution, features, test_edges, test_labels))

