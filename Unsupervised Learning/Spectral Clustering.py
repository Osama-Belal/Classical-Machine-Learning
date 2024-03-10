# Import libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph




def kways_normalized_cut_RBF(points, k, gamma_):
    A = rbf_kernel(points, gamma=gamma_)
    draw_graph(A)
    D_inv = np.diag(1 / np.sum(A, axis=1))
    La = np.eye(len(points)) - np.dot(D_inv, A)
    eigvals, eigvecs = np.linalg.eigh(La)
    indices = np.argsort(eigvals)[:k]
    Y = eigvecs[:, indices]

    Y_normalized = Y  # / np.linalg.norm(Y, axis=0, keepdims=True)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Y_normalized)
    cluster = kmeans.labels_

    return cluster, Y_normalized


def RBF_similarity(X, gamma_values):
    # Perform K-ways normalized cut for each gamma value
    for gamma in gamma_values:
        labels_, Y_ = kways_normalized_cut_RBF(X, k=3, gamma_=gamma)
        cluster_colors = ["r", "g", "b"]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(Y_)):
            ax.scatter(X[i][0], X[i][1], c=cluster_colors[labels_[i]])

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'RBF Similarity measure Gamma = {gamma}')
        plt.show()


def kways_normalized_cut_3_CNN(adj_matrix, k):
    A = adj_matrix
    draw_graph(A)
    D_inv = np.diag(1 / np.sum(A, axis=1))
    La = np.eye(len(adj_matrix)) - np.dot(D_inv, A)
    eigvals, eigvecs = np.linalg.eigh(La)
    indices = np.argsort(eigvals)[:k]
    Y = eigvecs[:, indices]

    Y_normalized = Y  # / np.linalg.norm(Y, axis=0, keepdims=True)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Y_normalized)
    clusters = kmeans.labels_

    return clusters, Y_normalized


def _3_NN_similarity(points):
    A = kneighbors_graph(points, n_neighbors=3, mode='connectivity', include_self=False)
    adjacency_matrix = A.toarray()

    clusters, Y_ = kways_normalized_cut_3_CNN(adjacency_matrix, k=3)
    cluster_colors = ["r", "g", "b"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(Y_)):
        ax.scatter(points[i][0], points[i][1], c=cluster_colors[clusters[i]])
        # ax.scatter(Y_[i][0], Y_[i][1], Y_[i][2], c=cluster_colors[clusters[i]])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('3_NN similarity measure')
    plt.show()


def draw_graph(adjacency_matrix):
    e = 1e-16
    labels = [chr(ord('a') + j) for j in range(len(adjacency_matrix))]

    gr = nx.Graph()
    for m in range(len(adjacency_matrix)):
        for j in range(m + 1, len(adjacency_matrix)):
            if adjacency_matrix[m][j] > e:
                gr.add_edge(m, j, weight=adjacency_matrix[m][j])
    pos = nx.spring_layout(gr)

    max_weight = max([d['weight'] for _, _, d in gr.edges(data=True)])
    for u, v, data in gr.edges(data=True):
        width = data['weight'] / max_weight
        nx.draw_networkx_edges(gr, pos, node_size=500, node_shape="o", edgelist=[(u, v)], width=width)

    nx.draw_networkx_labels(gr, pos)
    plt.show()


# Test
points = np.array([
    [5, 8], [10, 8], [11, 8],
    [6, 7], [10, 7], [12, 7], [13, 7],
    [5, 6], [10, 6], [13, 6], [14, 6],
    [6, 5], [11, 5], [15, 5],
    [2, 4], [3, 4], [5, 4], [6, 4], [7, 4], [9, 4], [15, 4],
    [3, 3], [7, 3], [8, 2]
])

RBF_similarity(points, [0.01, 0.1, 1, 10])
_3_NN_similarity(points)
