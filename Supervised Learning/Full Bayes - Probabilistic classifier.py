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
    covariances = np.zeros((n_clusters, X.shape[1]))
    for i, cluster in enumerate(clusters):
        covariances[i] = np.var(X[labels == cluster], axis=0)
    return covariances

def calculate_covariances_per_cluster(X, labels):
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    covariances = np.zeros((n_clusters, X.shape[1], X.shape[1]))
    for i, cluster in enumerate(clusters):
        covariances[i] = np.cov(X[labels == cluster], rowvar=False)
    return covariances

def calculate_priors_per_cluster(labels):
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    priors = np.zeros(n_clusters)
    for i, cluster in enumerate(clusters):
        priors[i] = np.sum(labels == cluster) / len(labels)
    return priors

def calculate_gaussian_multivariate_likelihood(x, mean, covariance):
    n = len(x)
    exponent = -0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean)
    return 1 / (np.sqrt((2 * np.pi) ** n * np.linalg.det(covariance))) * np.exp(exponent)

def calculate_gaussian_variate_likelihood(x, mean, variance):
    exponent = -0.5 * ((x - mean) ** 2 / variance)
    return 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(exponent)


# ------------------------------------------------------------------------------------------------

def classify_naive_bayes(x, means, variances, priors):
    n_clusters = means.shape[0]
    probabilities = np.zeros(n_clusters)

    for i in range(n_clusters):
        probabilities[i] = np.prod(calculate_gaussian_variate_likelihood(x, means[i], variances[i])) * (priors[i])
    return np.argmax(probabilities)

def classify_full_bayes(x, means, covariances, priors):
    n_clusters = means.shape[0]
    probabilities = np.zeros(n_clusters)

    for i in range(n_clusters):
        probabilities[i] = calculate_gaussian_multivariate_likelihood(x, means[i], covariances[i]) * (priors[i])
    return np.argmax(probabilities)

# ------------------------------------------------------------------------------------------------
# Test the classifier

X = np.array([
    [10, 1],
    [6 , 2],[7 , 2],[9 , 2],[12, 2],
    [3 , 3],[4 , 3],[5 , 3],[7 , 3],[9 , 3],[10, 3],[11, 3],
    [4 , 4],[6 , 4],[7 , 4],[8 , 4],[11, 4],
    [3 , 5],[4 , 5],[5 , 5],[7 , 5],[10, 5],[13, 5],
    [2 , 6],[3 , 6],
])

labels = np.array([2,0,2,0,2,1,1,0,1,0,2,2,0,0,1,0,2,0,1,1,1,2,2,0,0])


m = calculate_means_per_cluster(X, labels)
v = calculate_variances_per_cluster(X, labels)
cov = calculate_covariances_per_cluster(X, labels)
p = calculate_priors_per_cluster(labels)

print("Means for each cluster:")
print(m)
print("")
print("Covariances for each cluster:")
print(v)
print("")
print("Priors for each cluster:")
print(p)
print("")
print("Classify using Naive Bayes:")
print("Point [6, 5] belongs to C", classify_naive_bayes(np.array([6, 5]), m, v, p)+1)
print("Point [9, 4] belongs to C", classify_naive_bayes(np.array([9, 4]), m, v, p)+1)
print("Point [8, 5] belongs to C", classify_naive_bayes(np.array([8, 5]), m, v, p)+1)
print("")
print("Classify using Full Bayes:")
print("Point [6, 5] belongs to C", classify_full_bayes(np.array([6, 5]), m, cov, p)+1)
print("Point [9, 4] belongs to C", classify_full_bayes(np.array([9, 4]), m, cov, p)+1)
print("Point [8, 5] belongs to C", classify_full_bayes(np.array([8, 5]), m, cov, p)+1)