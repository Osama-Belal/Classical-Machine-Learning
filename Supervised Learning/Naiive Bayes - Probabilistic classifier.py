import numpy as np

def calculate_means_per_cluster(X, labels):
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    means = np.zeros((n_clusters, X.shape[1]))
    for i, cluster in enumerate(clusters):
      means[i] = np.mean(X[labels == cluster], axis=0)
    return means

def calculate_variances_per_cluster(X, labels):
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    variances = np.zeros((n_clusters, X.shape[1]))
    for i, cluster in enumerate(clusters):
        variances[i] = np.var(X[labels == cluster], axis=0)
    return variances

def calculate_priors_per_cluster(labels):
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    priors = np.zeros(n_clusters)
    for i, cluster in enumerate(clusters):
        priors[i] = np.sum(labels == cluster) / len(labels)
    return priors

def calculate_gaussian_likelihood(x, mean, variance):
    likelihood = []
    for i in range(len(mean)) :
      if variance[i] == 0 :
        if x[i] == mean[i] :
          likelihood.append(1)
        else :
          likelihood.append(0)
      else :
        likelihood.append(
            -1 * np.log(np.sqrt((2 * np.pi * variance[i]))) -
            (((x[i] - mean[i]) ** 2) / (2 * variance[i]))
                         )
    return np.sum(likelihood)

def classify(x, means, variances, priors):
    n_clusters = means.shape[0]
    probabilities = np.zeros(n_clusters)

    for i in range(n_clusters):
        probabilities[i] = calculate_gaussian_likelihood(x, means[i], variances[i]) + np.log(priors[i])
    return np.argmax(probabilities)

# ------------------------------------------------------------------------------------------------
# Calculate the confusion matrix and the classification accuracy

def calculate_confusion_matrix(X, labels, means, variances, priors):
    n_clusters = means.shape[0]
    confusion_matrix = np.zeros((n_clusters, n_clusters))

    errors = []
    for i in range(len(X)):
        predicted = classify(X[i], means, variances, priors) + 1
        confusion_matrix[int(labels[i] - 1), predicted - 1] += 1
        if confusion_matrix[int(labels[i] - 1), predicted - 1] == 1 and labels[i] != predicted:
            errors.append(X[i])

    correct = 0
    for i in range(len(confusion_matrix)):
        correct += confusion_matrix[i, i]

    return confusion_matrix, np.array(errors), correct / len(X)


# ------------------------------------------------------------------------------------------------
# Test the classifier

X = np.array([
    [1, 2],
    [3, 12],
    [5, 25],
    [7, 40],
    [9, 60],
    [11, 90]])
labels = np.array([0, 0, 1, 2, 1, 2])


m = calculate_means_per_cluster(X, labels)
v = calculate_variances_per_cluster(X, labels)
p = calculate_priors_per_cluster(labels)

confusion_matrix, errors, accuracy = calculate_confusion_matrix(X, labels, m, v, p)