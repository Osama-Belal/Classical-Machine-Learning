# Import libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

def max_matching(labels_true, labels_pred):
    pred_cluster_indices = {}
    for idx, label in enumerate(labels_pred):
        if label not in pred_cluster_indices:
            pred_cluster_indices[label] = []
        pred_cluster_indices[label].append(idx)

    true_clusters_matrix = np.zeros((len(pred_cluster_indices), len(pred_cluster_indices)))
    for i, cluster in enumerate(pred_cluster_indices.values()):
        true_cluster_counts = np.bincount([labels_true[i] for i in cluster])
        # make true_cluster_counts the same length as pred_cluster_indices
        true_cluster_counts = np.pad(true_cluster_counts, (0, len(pred_cluster_indices) - len(true_cluster_counts)))
        true_clusters_matrix[i, :] = true_cluster_counts

    # Calculate the maximum sum of any diagonal
    max_sum = 0
    for i in range(len(true_clusters_matrix)):
        sum = 0
        for j in range(len(true_clusters_matrix)):
            sum += true_clusters_matrix[(i+j)%len(true_clusters_matrix), j]
        max_sum = np.max([max_sum, sum])
    return max_sum / len(labels_true)



# Test maximum matching example

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


cl1 = _3_NN_similarity(points, 3)
cl2 = k_means(points, 3)

print("Maximum Matching of 3_NN_similarity: ", max_matching(ground_truth, cl1))
print("Maximum Matching of K-means: ", max_matching(ground_truth, cl2))