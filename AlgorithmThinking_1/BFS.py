"""
Code for BFS
"""

from collections import deque
import random

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
    largest_size = float('-inf')
    for component in connected_components:
        size = len(component)
        if size > largest_size:
            largest_size = size
    return largest_size