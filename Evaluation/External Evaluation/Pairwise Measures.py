# Import libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

def pairwise_measures(labels_true, labels_pred):
    # Mapping predicted clusters to indices
    pred_cluster_counts = {}
    for idx, label in enumerate(labels_pred):
        if label not in pred_cluster_counts:
            pred_cluster_counts[label] = np.zeros(len(np.unique(labels_true)), dtype=int)
        pred_cluster_counts[label][labels_true[idx]] += 1

    TP = 0
    FP = 0
    for cluster in pred_cluster_counts.values():
        TP += np.sum([n * (n-1)/2 for n in cluster])
        FP += np.sum([n * (np.sum(cluster) - n) for n in cluster])

    TN = 0
    FN = 0
    for Ci in pred_cluster_counts.values():
        for Cj in pred_cluster_counts.values():
            if not np.array_equal(Ci, Cj):
                TN += np.sum([Ci[k] * Cj[l] for k in range(len(Ci)) for l in range(len(Cj)) if k != l])
                FN += np.sum([Ci[k] * Cj[k] for k in range(len(Ci))])

    return TP, FP / 2, TN / 2, FN / 2

def Jaccard_index(labels_true, labels_pred):
    TP, FP, TN, FN = pairwise_measures(labels_true, labels_pred)
    return TP / (TP + FP + FN)

def Rand_index(labels_true, labels_pred):
    TP, FP, TN, FN = pairwise_measures(labels_true, labels_pred)
    return (TP + TN) / (TP + FP + TN + FN)


#  Test pairwise measures

points = np.array([
    [5, 8], [10, 8], [11, 8],
    [6, 7], [10, 7], [12, 7], [13, 7],
    [5, 6], [10, 6], [13, 6], [14, 6],
    [6, 5], [11, 5], [15, 5],
    [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [9, 4], [15, 4],
    [3, 3], [7, 3], [8, 2]
])

ground_truth = [
    1,2,2,
    1,2,2,2,
    1,2,2,2,
    1,2,2,
    0,0,1,1,1,1,2,
    0,1,1]

K = [2, 3, 4]



def plot_adjacency_matrix(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=200, font_size=10)
    plt.show()


def kways_normalized_cut_K_CNN(adj_matrix, c):
    A = adj_matrix
    plot_adjacency_matrix(A)

    D_inv = np.diag(1 / np.sum(A, axis=1))
    La = np.eye(len(adj_matrix)) - np.dot(D_inv, A)
    eigvals, eigvecs = np.linalg.eigh(La)
    indices = np.argsort(eigvals)[:c]
    Y = eigvecs[:, indices]

    Y_normalized = Y  # / np.linalg.norm(Y, axis=0, keepdims=True)
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(Y_normalized)
    clusters = kmeans.labels_

    return clusters, Y_normalized


def _3_NN_similarity(points, K):
    A = kneighbors_graph(points, n_neighbors=3, mode='connectivity', include_self=False)
    adjacency_matrix = A.toarray()

    clusters, Y_ = kways_normalized_cut_K_CNN(adjacency_matrix, c=K)
    cluster_colors = ["r", "g", "b", "y"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(Y_)):
        ax.scatter(points[i][0], points[i][1], c=cluster_colors[clusters[i]])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('3_NN similarity measure K = ' + str(K) + ' clusters')
    plt.show()
    return clusters


def k_means(points, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    clusters = kmeans.labels_

    cluster_colors = ["r", "g", "b", "y"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(points)):
        ax.scatter(points[i][0], points[i][1], c=cluster_colors[clusters[i]])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('K-means clustering K = ' + str(k))
    plt.show()
    return clusters


for C in K:
    cl1 = _3_NN_similarity(points, C)
    cl2 = k_means(points, C)

    print("K = ", C, "clusters ------------------------------------------------------")
    print("Jaccard Index of 3_NN_similarity: ", Jaccard_index(ground_truth, cl1))
    print("Jaccard Index of K-means: ", Jaccard_index(ground_truth, cl2))
    print("")
    print("Rand Index of 3_NN_similarity: ", Rand_index(ground_truth, cl1))
    print("Rand Index of K-means: ", Rand_index(ground_truth, cl2))
    print("")
