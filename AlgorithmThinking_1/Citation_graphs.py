"""
Application
"""

"""
Provided code for Application portion of Module 1

Imports physics citation graph
"""

# general imports
import urllib2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import operator as op
import time

###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

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
    total_edges = 0
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1: -1]:
            answer_graph[node].add(int(neighbor))
            total_edges += 1
    print 'total edges: ', total_edges
    return answer_graph

def compute_in_degrees(digraph):
    """
    compute in-degrees of a directed graph
    :param digraph: directed graph
    :return: dictionary {node: indegrees}
    """
    nodes_indegree = {}
    total_indegrees = 0
    for edges in digraph.itervalues():
        for edge_head in edges:
            if edge_head in nodes_indegree:
                nodes_indegree[edge_head] += 1
            else:
                nodes_indegree[edge_head] = 1

    # for node, edges in digraph.items():
    #     k = len(edges)
    #     nodes_degree[node] = k
    #     total_indegrees += k
    # print 'total_indegrees: ', total_indegrees
    return nodes_indegree

def in_degree_distribution(digraph):
    """
    computer distribution of in-degrees
    :param digraph: directed graph
    :return:dictionary
    """
    degree_distribution = {}
    nodes_degree = compute_in_degrees(digraph)
    for degree in nodes_degree.itervalues():
        if degree in degree_distribution:
            degree_distribution[degree] += 1
        else:
            degree_distribution[degree] = 1
    return degree_distribution

def plot_distribuition(distribution, title):
    # sorted(student_tuples, key=lambda student: student[2])
    #print distribution
    sum = float(reduce(lambda x, y: x+y, distribution.itervalues()))
    #print sum
    x = distribution.keys()
    y = [v for v in distribution.itervalues()]
    combine = [[],[]]
    combine[0] = x
    combine[1] = y
    combine = np.asarray(combine)
    combine = combine[:, np.argsort(combine[0])]
    x=combine[0]
    y=combine[1]
    # print combine
    y.astype(float)
    y = y/sum
    # print np.sum(y)
    #
    # #print x, np.amin(x), np.amax(x)
    # #print y, y.shape
    plt.plot(x, y, '.')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('in-Degrees')
    plt.ylabel('Frequency')
    # plt.xticks(range(1, np.amax(x)+1))
    plt.xlim(1, np.amax(x)*2)
    plt.grid(True)
    plt.title('Log-Log plot of degree distribution for ' + title)
    plt.show()

def plot_citationgraph_distribution():
    citation_graph = load_graph(CITATION_URL)
    degree_distrbtn = in_degree_distribution(citation_graph)
    plot_distribuition(degree_distrbtn, 'citation graph')


def er_graph(num_node, probability):
    graph = {}
    for node in range(num_node):
        edges = filter(
            lambda nd: random.random() < probability
                       and nd != node, range(num_node))
        graph[node] = set(edges)
    return graph

def er_distribution(num_node, probability):
    distribution = {}
    def calc_subset(n,k):
        return math.factorial(n)/(math.factorial(k) * math.factorial(n - k))

    def ncr(n, r):
        r = min(r, n - r)
        numer = reduce(op.mul, xrange(n, n - r, -1), 1)
        denom = reduce(op.mul, xrange(1, r + 1), 1)
        return numer // denom
    for k in range(num_node):
        num_subset = ncr(num_node - 1, k)
        distribution[k] = \
            (num_subset * (probability ** k) * ((1 - probability) ** (num_node - 1 - k))) \
            * num_node
    return distribution

def plot_er_distribution():
    ergraph = er_graph(2770, 0.1)
    degree_distrbtn = in_degree_distribution(ergraph)
    plot_distribuition(degree_distrbtn)

"""
Q2:
1. yes. Expected value of in-degree is the same for every node, which is (n-1) * p
2. The in-degree distribution is a bell-shape convex curve with its opening facing down
3. No. The two distribution look quite differently.Citation graph's distribution is more
of a monotonic decreasing curve, while the er graph distribution is more like a symetric
curve (skewed to one side)

Q3.
Total # of nodes: 27770
Total # of edges: 352,768
average # of edges per node: 12.7
n = 27770
m = 13

Q4.
1. DPA graph is similar to that of the citation graph. They both are in shape of monotonic
decreasing curve and both have a flat tail at the end of the curve. They both have similar
range in x and y values.
2. The "rich gets richer" phenomenon mimics the behavior of the DPA process.
The first few nodes has higher probability to be selected by the random process which 
makes them always have higher probability for following node selection processes.
3. Still the  "rich gets richer" phenomenon.
The popularity of a particular paper causes new paper to cite the same choice 
(which can lead to the outsized influence of the first few papers).

"""


class DPATrial:
    """
    Simple class to encapsulate optimized trials for DPA algorithm

    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities

    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a DPATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]
        # print self._node_numbers
    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities

        Returns:
        Set of nodes
        """

        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))

        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))

        # update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

def dpa_algorithm(num_nodes, avg_edges):
    graph = {}
    dpa = DPATrial(avg_edges)
    for node in range(avg_edges):
        neighbors = [n for n in range(avg_edges)]
        del neighbors[node]
        graph[node] = set(neighbors)
    totindeg = (avg_edges - 1) ** 2
    for i in range(avg_edges, num_nodes):
        new_node_neighbors = dpa.run_trial(avg_edges)
        graph[i] = new_node_neighbors
    return graph

def plot_dpa_distribution():
    start = time.time()
    dpagraph = dpa_algorithm(27770, 13)
    end = time.time()

    # total_indegrees = 0
    # for node, edges in dpagraph.items():
    #     k = len(edges)
    #     total_indegrees += k
    # print 'total_indegrees: ', total_indegrees

    print 'running time:', end-start, 'seconds'
    # print dpagraph
    degree_distrbtn = in_degree_distribution(dpagraph)
    plot_distribuition(degree_distrbtn, 'dpa graph')

# plot_citationgraph_distribution()

# plot_dpa_distribution()