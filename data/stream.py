# find and cite data from http://networkrepository.com/dynamic.php
import networkx as nx

def load(path='data/fb-messages.edges', delimiter=' '):
    G = nx.Graph()
    with open(path) as file:
        for line in file:
            if line[0] == '%':
                continue
            edge = line[:-1].split(delimiter)
            if len(edge)<2:
                continue
            G.add_edge(edge[0], edge[1])
    return G