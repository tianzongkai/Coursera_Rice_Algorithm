"""
Code for Graph Resillence
week 3 application #2
"""

# general imports
from collections import deque
import urllib2
import random
import time
import math
import matplotlib.pyplot as plt

def bfs_visited(ugraph, start_node):
    """
    Implementation of bfs_visited algorithm
    :param ugraph: undirected graph
    :param start_node: a node to start
    :return: a set consisting of all connected components
    """
    # len(d)
    # d.append('1')
    # d.popleft()
    # d.pop() -- pop from right
    # d.extendleft([0])
    # d.extend([6,7,8]) -- extend from right
    queue = deque()
    visited = set([start_node])
    queue.append(start_node)
    while len(queue) > 0:
        popped = queue.popleft()
        for neighbor in ugraph[popped]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def cc_visited(ugraph):
    """
    Implementation for cc_visited algorithm
    :param ugraph: undirected graph
    :return: a list of sets, each set consists of all the nodes (and nothing else)
    in a connected component
    """
    remaining_nodes = ugraph.keys()
    connected_components = []
    while len(remaining_nodes) > 0:
        choiced = random.choice(remaining_nodes)
        visted_nodes = bfs_visited(ugraph, choiced)
        connected_components.append(visted_nodes)
        remaining_nodes = list(set(remaining_nodes) - visted_nodes)
    return connected_components

def largest_cc_size(ugraph):
    """
    :param ugraph: undirected graph
    :return:  the size (an integer) of the largest connected component in ugraph
    """
    connected_components = cc_visited(ugraph)
    largest_size = 0
    for component in connected_components:
        size = len(component)
        if size > largest_size:
            largest_size = size
    return largest_size

def compute_resilience(ugraph, attack_order):
    """
    remove these nodes (and their edges) from the graph one at a time
    and then measure the "resilience" of the graph at each removal by
    computing the size of its largest remaining connected component.
    :param ugraph: undirected graph
    :param attack_order: a list of nodes
    :return: a list whose k+1th entry is the size of the largest connected component
    in the graph after the removal of the first k nodes in attack_order.
    The first entry (indexed by zero) is the size of the largest connected component
     in the original graph
    """

    sizes_after_removal = [largest_cc_size(ugraph)]
    for attack_node in attack_order:
        # print attack_node
        if attack_node in ugraph:
            del ugraph[attack_node]
            # print ugraph
            for node, edges in ugraph.iteritems():
                if attack_node in edges:
                    edges.remove(attack_node)
                    ugraph[node] = edges
            # print ugraph
        sizes_after_removal.append(largest_cc_size(ugraph))
    return sizes_after_removal


############################################
# Provided code
def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)

def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree

    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)

    order = []
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node

        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order

##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph

    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[: -1]

    print "Loaded graph with", len(graph_lines), "nodes"

    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1: -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph

def compute_degrees(ugraph):
    """
    compute degrees of a directed graph
    :param ugraph: undirected graph
    :return: dictionary {node: degrees}
    """
    nodes_degree = {}
    total_degrees = 0
    for edges in ugraph.itervalues():
        for edge_head in edges:
            if edge_head in nodes_degree:
                nodes_degree[edge_head] += 1
            else:
                nodes_degree[edge_head] = 1

    for degree in nodes_degree.itervalues():
        total_degrees += degree
    total_degrees /= 2
    print 'total_degrees: ', total_degrees
    return nodes_degree

def create_er_graph(num_node, probability):
    graph = {}
    for node in range(num_node):
        edges = filter(
            lambda nd: random.random() < probability
                       and nd != node, range(num_node))
        graph[node] = set(edges)
    return graph

class UPATrial:
    """
    Simple class to encapsulate optimizated trials for the UPA algorithm

    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities

    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]

    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that each node number
        appears in correct ratio

        Returns:
        Set of nodes
        """

        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))

        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))

        # update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

def upa_algorithm(num_nodes, avg_edges):
    graph = {}
    upa = UPATrial(avg_edges)
    for node in range(avg_edges):
        neighbors = [n for n in range(avg_edges)]
        del neighbors[node]
        graph[node] = set(neighbors)
    # print graph
    # totindeg = (avg_edges - 1) ** 2
    for i in range(avg_edges, num_nodes):
        new_node_neighbors = upa.run_trial(avg_edges)
        graph[i] = set(new_node_neighbors)

        #add the node as a neighbor to each node in new_node_neighbors
        for neighbor in new_node_neighbors:
            graph[neighbor].add(i)
        # print graph
    return graph

def random_order(ugraph):
    nodes_list = ugraph.keys()
    random.shuffle(nodes_list)
    return nodes_list

"""
Q1: a standard line plot for each curve, not log/log. 
    * x-axis: # of nodes removed (ranging from zero to # of nodes in the graph)
    * y-axis: size of the largest connect component in the graphs resulting from 
      the node removal
    * The text labels in this legend should include the values for p and m 
      that you used in computing the ER and UPA graphs, respectively
Q2: 20% of nodes are about 250 nodes. So there'are ~990 remaining nodes,
    75% of which is ~750. As long as the size of largest cc is above 750 when
    x-axis = 250, the graph is resilient. At 250, all three graphs are above 750,
    so they're all resilient under the attack.
Q3: 
"""

def plot_resilience():
    # Loaded graph with 1239 nodes
    # total_degrees:  3047
    # num of edges for a complete graph of 1239 nodes = 1239 * 1238 / 2 = 766,941
    # p = 3047 / 766941 = 0.04
    # avg degree = 2.5

    network_graph = load_graph(NETWORK_URL)
    network_graph_random_order = random_order(network_graph)
    network_graph_resilience = compute_resilience(network_graph, network_graph_random_order)

    # compute_degrees(network_graph)

    num_nodes_removed = range(1240)

    er_graph = create_er_graph(1239, 0.004)
    print '\ner graph:'
    compute_degrees(er_graph)
    print er_graph
    er_graph_random_order = random_order(er_graph)
    er_graph_resilience = compute_resilience(er_graph, er_graph_random_order)

    upa_graph = upa_algorithm(1239, 3)
    print '\nupa graph'
    compute_degrees(upa_graph)
    print upa_graph
    upa_graph_random_order = random_order(upa_graph)
    upa_graph_resilience = compute_resilience(upa_graph, upa_graph_random_order)

    plt.plot(num_nodes_removed, network_graph_resilience, label="network graph")
    plt.plot(num_nodes_removed, er_graph_resilience, label="er graph, p = 0.004")
    plt.plot(num_nodes_removed, upa_graph_resilience, label="upa graph, m = 3")
    plt.legend()
    plt.xlim(0,1300)
    plt.ylim(0,1300)
    plt.xticks(range(0, 1301, 100))
    plt.yticks(range(0, 1301, 100))
    plt.grid()
    plt.ylabel('Size of largest connect compnent')
    plt.xlabel('Number of nodes removed')

    plt.title("Graph Resilience")

    plt.show()

# plot_resilience()
# upa_graph = upa_algorithm(1239, 3)


###Q4###
def fast_targeted_order(graph):
    n = len(graph)
    degree_sets = [set() for _ in range(n)]
    for i in range(n):
        d = len(graph[i])
        degree_sets[d].add(i)
    l = []
    i = 0
    for k in range(n - 1, -1, -1):
        while degree_sets[k]:
            u = random.choice(degree_sets[k])

