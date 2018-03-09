"""
Application
"""

import urllib2

"""
Provided code for Application portion of Module 1

Imports physics citation graph
"""

# general imports
import urllib2
import numpy as np
import matplotlib.pyplot as plt

# Set timeout for CodeSkulptor if necessary
# import codeskulptor
# codeskulptor.set_timeout(20)


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

    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1: -1]:
            answer_graph[node].add(int(neighbor))
    return answer_graph

def compute_in_degrees(digraph):
    """
    compute in-degrees of a directed graph
    :param digraph: directed graph
    :return: dictionary
    """
    nodes_degree = {}
    for node in digraph.keys():
        nodes_degree[node] = 0
    for neighbors in digraph.itervalues():
        for node in neighbors:
            nodes_degree[node] += 1
    return nodes_degree

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

def plot_distribuition(distribution):
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
    plt.xlabel('Degrees')
    plt.ylabel('Frequency')
    plt.title('Log-Log plot of degree distribution')
    plt.show()


citation_graph = load_graph(CITATION_URL)
degree_distrbtn = in_degree_distribution(citation_graph)
plot_distribuition(degree_distrbtn)

