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
    Q = deque()
    visited = set([start_node])
    Q.append(start_node)
    while len(Q) > 0:
        j = Q.popleft()
        for neighbor in ugraph[j]:
            if neighbor not in visited:
                visited.add(neighbor)
                Q.append(neighbor)
    return visited

def cc_visited(ugraph):
    RemainingNodes = ugraph.keys()
    CC = set()
    while len(RemainingNodes) > 0:
        i = random.choice(RemainingNodes)
        W = bfs_visited(ugraph, i)
        CC.union(W)
        RemainingNodes = list(set(RemainingNodes) - W)
    return CC