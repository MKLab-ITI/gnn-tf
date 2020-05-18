import networkx as nx


def load(path, delimiter=' '):
    G = nx.Graph()
    with open(path) as file:
        for line in file:
            if line[0] == '#':
                continue
            edge = line[:-1].split(delimiter)
            if len(edge)!=2:
                continue
            G.add_edge(*edge)
    return G


def nutella():
    return nx.ego_graph(load('data/p2p-Gnutella04.txt', '\t'), '0', 4)


def facebook():
    # http://snap.stanford.edu/data/ego-Facebook.html
    return load('data/facebook_combined.txt')


def twitter():
    # https://snap.stanford.edu/data/ego-Twitter.html
    return nx.ego_graph(load('data/twitter_combined.txt'), '214328887', 1)