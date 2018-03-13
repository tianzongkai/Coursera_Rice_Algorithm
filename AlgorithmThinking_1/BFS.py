"""
Code for BFS
"""

from collections import deque
import random

def bfs_visited(ugraph, start_node):
    """
    len(d)
    d.append('1')
    d.popleft()
    d.pop() -- pop from right
    d.extendleft([0])
    d.extend([6,7,8]) -- extend from right
    """
    q = deque()
    visited = set([start_node])
    q.append(start_node)
    while len(q) > 0:
        j_ = q.popleft()
        for neighbor in ugraph[j_]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return visited

def cc_visited(ugraph):
    """
    Implementation for cc_visited algorithm
    :param ugraph: undirected graph
    :return: a list of sets, each set consists of all the nodes (and nothing else)
    in a connected component
    """
    remaining_nodes = ugraph.keys()
    c_c = []
    while len(remaining_nodes) > 0:
        i_ = random.choice(remaining_nodes)
        w_ = bfs_visited(ugraph, i_)
        c_c.append(w_)
        remaining_nodes = list(set(remaining_nodes) - w_)
    return c_c