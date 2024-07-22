import numpy as np

def compute_eigenvalues_eigenvectors(matrix):
    matrix = np.array(matrix)
    return np.linalg.eig(matrix)

# Example
matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(matrix)
print(eigenvectors) # Output: [[ 0.89442719 -0.70710678] [ 0.4472136   0.70710678]]