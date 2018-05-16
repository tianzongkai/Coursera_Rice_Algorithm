import math
import random
import alg_cluster
import time
import alg_project3_solution as student
import matplotlib.pyplot as plt

def gen_random_clusters(num_clusters):
    return [alg_cluster.Cluster(set([]),random.uniform(-1, 1),random.uniform(-1, 1),0,0 )
            for _ in range(num_clusters)]

def plot_running_time():
    slow_runnint_time = []
    fast_running_time = []
    x_axis = range(2,201)
    for num_clusters in x_axis:
        cluster_list = gen_random_clusters(num_clusters)
        cluster_list.sort(key=lambda cluster: cluster.horiz_center())
        start = time.clock()
        student.slow_closest_pair(cluster_list)
        end_slow = time.clock()
        student.fast_closest_pair(cluster_list)
        end_fast = time.clock()
        slow_runnint_time.append(end_slow-start)
        fast_running_time.append(end_fast-end_slow)

    plt.plot(x_axis, slow_runnint_time, label="slow")
    plt.plot(x_axis, fast_running_time, label="fast")
    plt.legend()
    plt.ylabel('Running time')
    plt.xlabel('Number of clusters')
    plt.title("Closest Pair Algorithms Comparison")
    plt.show()


####Efficiency####
# Question 1
# plot_running_time()


# Question 7
def compute_distortion(cluster_list, data_table):
    distortion_list = [cluster.cluster_error(data_table) for cluster in cluster_list]
    distortion = sum(distortion_list)
    return distortion
