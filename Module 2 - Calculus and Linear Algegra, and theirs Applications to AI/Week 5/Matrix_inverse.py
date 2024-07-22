import numpy as np

def inverse_matrix(matrix):
    matrix = np.array(matrix)
    return np.linalg.inv(matrix)

# Example
m1 = np.array([[-2, 6], [8,-4]])
result = inverse_matrix(m1)
print(result) # Output: [[ 0.1 0.15] [0.2 0.05]]