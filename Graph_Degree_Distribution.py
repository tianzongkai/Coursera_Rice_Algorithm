EX_GRAPH0 = {0: set([1,2]),
             1: set([]),
             2: set([])}

EX_GRAPH1 = {0: set([1,4,5]),
             1: set([2,6]),
             2: set([3]),
             3: set([0]),
             4: set([1]),
             5: set([2]),
             6: set([])}

EX_GRAPH2 = {0: set([1,4,5]),
             1: set([2,6]),
             2: set([3,7]),
             3: set([7]),
             4: set([1]),
             5: set([2]),
             6: set([]),
             7: set([3]),
             8: set([1,2]),
             9: set([0,3,4,5,6,7])}


def make_complete_graph(num_nodes):
    graph = {}
    for node in range(num_nodes):
        neighbors = [n for n in range(num_nodes)]
        del neighbors[node]
        graph[node] = set(neighbors)
    return graph

def compute_in_degrees(digraph):
    nodes_degree = {}
    for node in range(len(digraph)):
        nodes_degree[node] = 0
    for neighbors in digraph.itervalues():
        for node in neighbors:
            nodes_degree[node] += 1
    return nodes_degree

#print compute_in_degrees(EX_GRAPH2)
