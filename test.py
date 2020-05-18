from data.pairs import twitter as load
import numpy as np
import gnn_dense as gnn
import numpy.random
import utils

measure_values = dict()
for _ in range(5):
    print("Loading...")
    G = load()
    print(len(G), "nodes,", G.number_of_edges(), "edges")
    features = np.identity(len(G)).astype('float32')

    print("Creating training/test data...")
    adjacency = gnn.get_adjacency(G)
    edges = gnn.negative_sampling(adjacency)
    print('Total edges (including negative samples)', edges.shape[0])
    training_ids = numpy.random.choice(list(range(edges.shape[0])), edges.shape[0]*8//10)
    training_edges = edges[training_ids,:]
    training_labels = gnn.get_labels(adjacency, training_edges)

    if len(training_edges) != len(edges):
        test_edges = edges[np.setdiff1d(edges, training_ids),:]
        test_labels = gnn.get_labels(adjacency, test_edges)
        convolution = gnn.get_convolution(adjacency, test_edges)
    else:
        convolution = gnn.get_convolution(adjacency)

    print("Creating model...")
    model = gnn.PageRank(0.5)
    #model = gnn.Model(len(G), [32, 32, 32], activation="linear")
    #model = gnn.Random()

    print("Training...")
    model.train(convolution, features, training_edges, training_labels, epochs=150)
    print('Testing...')

    predicted_labels = model.predict(convolution, features, test_edges)
    for measure in ['AUC']:
        if measure not in measure_values:
            measure_values[measure] = list()
        #print('Training', measure, utils.evaluate(training_labels, model.predict(convolution, features, training_edges), measure, training_edges))
        if len(training_edges) != len(edges):
            value = utils.evaluate(test_labels, predicted_labels, measure, test_edges)
            measure_values[measure].append(value)
            print('Test', measure, value)

for measure in measure_values:
    print('Avg test', measure, sum(measure_values[measure])/len(measure_values[measure]))