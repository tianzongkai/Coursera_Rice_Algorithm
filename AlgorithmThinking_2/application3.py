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

# Question 10
DIRECTORY = "http://commondatastorage.googleapis.com/codeskulptor-assets/"
DATA_3108_URL = DIRECTORY + "data_clustering/unifiedCancerData_3108.csv"
DATA_896_URL = DIRECTORY + "data_clustering/unifiedCancerData_896.csv"
DATA_290_URL = DIRECTORY + "data_clustering/unifiedCancerData_290.csv"
DATA_111_URL = DIRECTORY + "data_clustering/unifiedCancerData_111.csv"
DATA_24_URL = DIRECTORY + "data_clustering/unifiedCancerData_24.csv"


def load_data_table(data_url):
    """
    Import a table of county-based cancer risk data
    from a csv format file
    """
    import urllib2
    data_file = urllib2.urlopen(data_url)
    data = data_file.read()
    data_lines = data.split('\n')
    print "Loaded", len(data_lines), "data points"
    data_tokens = [line.split(',') for line in data_lines]
    return [[tokens[0], float(tokens[1]), float(tokens[2]), int(tokens[3]), float(tokens[4])]
            for tokens in data_tokens]

def clustering():
    title_list = ['111 counties', '290 counties', '896 counties']
    url_list = [DATA_111_URL,DATA_290_URL,DATA_896_URL]
    distortion_hierarchical = [[],[],[]]
    distortion_kmeans = [[],[],[]]
    num_clusters_list = range(20, 5, -1)

    for idx in range(len(url_list)):
        data_table = load_data_table(url_list[idx])
        cluster_list = []
        for line in data_table:
            cluster_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        cluster_list_copy = [cluster.copy() for cluster in cluster_list]
        for num_cluster in num_clusters_list:
            cluster_list = student.hierarchical_clustering(cluster_list, num_cluster)
            distortion = compute_distortion(cluster_list, data_table)
            distortion_hierarchical[idx].append(distortion)
            print "Displaying", len(cluster_list), "hierarchical clusters, distortion:", distortion

        for num_cluster in num_clusters_list:
            cluster_list = student.kmeans_clustering(cluster_list_copy, num_cluster, 5)
            distortion = compute_distortion(cluster_list, data_table)
            distortion_kmeans[idx].append(distortion)
            print "Displaying", len(cluster_list), "k-means clusters, distortion:", distortion

        plot_num = 131 + idx
        plt.subplot(plot_num)
        plt.plot(num_clusters_list, distortion_hierarchical[idx], "o-", label="hierarchical")
        plt.plot(num_clusters_list, distortion_kmeans[idx], "x-", label="kmeans")
        plt.legend()
        plt.ylabel('Distortion')
        plt.xlabel('Number of clusters')
        plt.grid(True)
        plt.title(title_list[idx])
    plt.show()

clustering()


