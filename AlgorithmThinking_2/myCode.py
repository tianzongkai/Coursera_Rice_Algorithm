"""
Student template code for Project 3
Student will implement five functions:

slow_closest_pair(cluster_list)
fast_closest_pair(cluster_list)
closest_pair_strip(cluster_list, horiz_center, half_width)
hierarchical_clustering(cluster_list, num_clusters)
kmeans_clustering(cluster_list, num_clusters, num_iterations)

where cluster_list is a 2D list of clusters in the plane
"""

import math
import alg_cluster


######################################################
# Code for closest pairs of clusters

def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function that computes Euclidean distance between two clusters in a list

    Input: cluster_list is list of clusters, idx1 and idx2 are integer indices for two clusters

    Output: tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


def slow_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (slow)

    Input: cluster_list is the list of clusters

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    """
    ret_d = (float('inf'), -1, -1)
    for u in range(len(cluster_list)):
        for v in range(len(cluster_list)):
            if u != v:
                current_d = pair_distance(cluster_list, u, v)
                if current_d[0] < ret_d[0]:
                    ret_d = current_d

    return ret_d


def fast_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (fast)

    Input: cluster_list is list of clusters SORTED such that horizontal positions of their
    centers are in ascending order

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.
    """
    n = len(cluster_list)
    if n <= 3:
        ret_d = slow_closest_pair(cluster_list)
    else:
        ### divide ###
        m = n / 2
        cluster_l = cluster_list[0:m]
        cluster_r = cluster_list[m:n]
        ret_l = fast_closest_pair(cluster_l)
        ret_r = fast_closest_pair(cluster_r)

        ### merge ###
        if ret_l[0] <= ret_r[0]:
            ret_d = ret_l
        else:
            ret_d = (ret_r[0], ret_r[1] + m, ret_r + m)
        mid = (cluster_list[m-1].horiz_center() + cluster_list[m].horiz_center()) / 2
        ret_cps = closest_pair_strip(cluster_list, mid, ret_d[0])
        if ret_cps[0] < ret_d[0]:
            ret_d = ret_cps

    return ret_d


def closest_pair_strip(cluster_list, horiz_center, half_width):
    """
    Helper function to compute the closest pair of clusters in a vertical strip

    Input: cluster_list is a list of clusters produced by fast_closest_pair
    horiz_center is the horizontal position of the strip's vertical center line
    half_width is the half the width of the strip (i.e; the maximum horizontal distance
    that a cluster can lie from the center line)

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] lie in the strip and have minimum distance dist.
    """
    # [ num for num in numbers if num < 5 ]
    s = filter(lambda idx: cluster_list[idx].horiz_center() < half_width,
               range(len(cluster_list)))
    s.sort(key=lambda cluster: cluster.vert_center())
    k = len(s)
    ret_d = (float('inf'), -1, -1)
    for u in range(k-3):
        for v in range(u+1, min(u+3,k-1)):
            ret_s = pair_distance(cluster_list, s[u], s[v])
            if ret_s[0] < ret_d[0]:
                ret_d = ret_s
    return ret_d


















######################################################################
# Code for hierarchical clustering


def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function may mutate cluster_list

    Input: List of clusters, integer number of clusters
    Output: List of clusters whose length is num_clusters
    """

    return []


######################################################################
# Code for k-means clustering


def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters
    Note: the function may not mutate cluster_list

    Input: List of clusters, integers number of clusters and number of iterations
    Output: List of clusters whose length is num_clusters
    """

    # position initial clusters at the location of clusters with largest populations

    return []

