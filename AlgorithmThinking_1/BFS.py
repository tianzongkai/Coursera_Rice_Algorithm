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

# ugraph = {0: set([1,2]),
#           1: set([3]),
#           2: set([2]),
#           3: set([1,2,3]),
#           4: set([])
#           }
# attack_order = [1,2]
# print largest_cc_size(ugraph)
# print compute_resilience(ugraph, attack_order)
