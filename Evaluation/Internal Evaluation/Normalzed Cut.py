# Import libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

def create_proximity_matrix(X, labels):
    pred_cluster_indices = {}
    for idx, label in enumerate(labels):
        if label not in pred_cluster_indices:
            pred_cluster_indices[label] = []
        pred_cluster_indices[label].append(idx)

    proximity_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))

    intra_sum = 0
    inter_sum = 0
    Nin = 0
    Nout = 0
    for i, cluster in pred_cluster_indices.items():
        Wii = np.sum([np.linalg.norm(X[i] - X[j]) for i in cluster for j in cluster])
        proximity_matrix[i, i] = Wii
        intra_sum += Wii
        Nin += len(cluster) * (len(cluster) - 1) / 2

    for i, Ci in pred_cluster_indices.items():
        for j, Cj in pred_cluster_indices.items():
            if i != j:
                Wij = np.sum([np.linalg.norm(X[i] - X[j]) for i in Ci for j in Cj])
                proximity_matrix[i, j] = Wij
                proximity_matrix[j, i] = Wij
                inter_sum += Wij
                Nout += len(Ci) * len(Cj)
    return proximity_matrix, intra_sum, inter_sum, Nin, Nout

def NormalizedCut(X, labels):
    NC = 0
    proximity_matrix, intra_sum, inter_sum, Nin, Nout = create_proximity_matrix(X, labels)
    for i in range(len(proximity_matrix)):
        NC += (np.sum(proximity_matrix[i]) - proximity_matrix[i, i]) / np.sum(proximity_matrix[i])
    return NC


# Test the NormalizedCut function

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
    print("Normalized Cut for 3_NN similarity measure: ", NormalizedCut(points, cl1))
    print("Normalized Cut for K-means clustering: ", NormalizedCut(points, cl2))
