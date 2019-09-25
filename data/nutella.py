import networkx as nx


def load():
    G = nx.Graph()
    with open('data/p2p-Gnutella04.txt') as file:
        for line in file:
            if line[0] == '#':
                continue
            edge = line[:-1].split('\t')
            if len(edge)!=2:
                continue
            G.add_edge(*edge)
    return nx.ego_graph(G, '0', 3)