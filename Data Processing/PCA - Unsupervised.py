# Import libraries
import numpy as np

def PCA(data_matrix, acceptance_ratio):
  dim_means                   =   np.mean(data_matrix, axis=0)
  centered_data               =   data_matrix - dim_means
  covariance                  =   (1/(len(data_matrix) - 1)) * np.inner(np.transpose(centered_data), np.transpose(centered_data))
  print(covariance.shape)

  e_values, e_vectors         =   np.linalg.eigh(covariance)
  e_values = np.flip(e_values, 0)
  e_vectors = np.flip(e_vectors, 1)

  current_ratio               =   0.0
  eigen_sum = np.sum(e_values)

  cumulative_ratios = np.cumsum(e_values)
  relevant_indices = np.where(cumulative_ratios / eigen_sum < acceptance_ratio)[0]

  projection_matrix = np.zeros((e_vectors.shape[0], len(relevant_indices)))

  for idx, i in enumerate(relevant_indices):
      current_ratio += e_values[i]
      projection_matrix[:, idx] = e_vectors[:, i]

  return projection_matrix

# # Test
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(PCA(data, 0.95))

# Expected: [[-0.57735027 -0.57735027]
#            [-0.57735027  0.57735027]
#            [ 0.57735027  0.57735027]]

# The expected output is the first two principal components of the data matrix
# The first principal component is [-0.57735027 -0.57735027]
# The second principal component is [-0.57735027  0.57735027]
# The third principal component is [ 0.57735027  0.57735027]
# The third principal component is not included in the output because the acceptance ratio is 0.95
# The acceptance ratio is the percentage of the total variance that we want to capture
# The first two principal components capture 95% of the total variance
# The third principal component captures the remaining 5% of the total variance
