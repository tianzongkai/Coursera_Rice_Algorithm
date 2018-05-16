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
    for u_idx in range(len(cluster_list)):
        for v_idx in range(len(cluster_list)):
            if u_idx != v_idx:
                current_d = pair_distance(cluster_list, u_idx, v_idx)
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
    n_len = len(cluster_list)
    if n_len <= 3:
        ret_d = slow_closest_pair(cluster_list)
    else:
        ### divide ###
        mid = n_len / 2
        cluster_l = cluster_list[0:mid]
        cluster_r = cluster_list[mid:n_len]
        ret_l = fast_closest_pair(cluster_l)
        ret_r = fast_closest_pair(cluster_r)

        ### merge ###
        if ret_l[0] <= ret_r[0]:
            ret_d = ret_l
        else:
            ret_d = (ret_r[0], ret_r[1] + mid, ret_r[2] + mid)
        mid = (cluster_list[mid-1].horiz_center() + cluster_list[mid].horiz_center()) / 2
        ret_cps = closest_pair_strip(cluster_list, mid, ret_d[0])
        if ret_cps[0] < ret_d[0]:
            ret_d = ret_cps
    # ret_d :(dist, idx1, idx2)
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
    s_cluster = filter(lambda idx: abs(cluster_list[idx].horiz_center() - horiz_center) < half_width,
               range(len(cluster_list)))
    # print 'original s', s
    s_cluster.sort(key=lambda idx: cluster_list[idx].vert_center())
    # print 'sorted s', s
    k_len = len(s_cluster)
    # print 'k =', k
    ret_d = (float('inf'), -1, -1)
    for u_idx in range(k_len-1):
        for v_idx in range(u_idx+1, min(u_idx+3,k_len-1)+1):
            ret_s = pair_distance(cluster_list, s_cluster[u_idx], s_cluster[v_idx])
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
    cluster_list.sort(key=lambda cluster: cluster.horiz_center())
    # n = len(cluster_list)
    while (len(cluster_list) > num_clusters):
        _dest, i_idx, j_idx = fast_closest_pair(cluster_list)
        cluster_list[i_idx] = cluster_list[i_idx].merge_clusters(cluster_list[j_idx])
        cluster_list.remove(cluster_list[j_idx])
        cluster_list.sort(key=lambda cluster: cluster.horiz_center())

    return cluster_list


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

    n_len = len(cluster_list)
    list_copy = list(cluster_list)
    list_copy.sort(key=lambda cluster: cluster.total_population(), reverse=True)
    old_cluster = [cluster.copy() for cluster in list_copy[0:num_clusters]]
    # cluster_list.sort(key=lambda cluster: cluster.horiz_center())
    for _ in range(num_iterations):
        new_cluster = [alg_cluster.Cluster(set([]),0,0,0,0) for _ in range(num_clusters)]
        for j_idx in range(n_len):
            dist_list = [(cluster_list[j_idx].distance(old_cluster[f]), f) for f in range(num_clusters)]
            dist_list.sort(key=lambda dist_tuple: dist_tuple[0])
            argmin_closest = dist_list[0][1]
            new_cluster[argmin_closest].merge_clusters(cluster_list[j_idx])
        old_cluster = [c_i.copy() for c_i in new_cluster]
    return new_cluster

