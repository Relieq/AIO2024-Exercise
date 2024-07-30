import numpy as np

def compute_dot_product(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2)

# Example 1
v1 = np.array([0, 1, -1, 2])
v2 = np.array([2, 5, 1, 0])
result = compute_dot_product(v1, v2)
print(round(result, 2)) # Output: 4

def matrix_multi_vector(matrix, vector):
    matrix = np.array(matrix)
    vector = np.array(vector)
    return matrix.dot(vector)

# Example 2
m = np.array([[-1, 1, 1], [0,-4, 9]])
v = np.array([0, 2, 1])
result = matrix_multi_vector(m, v)
print(result) # Output: [ 3 1]

def matrix_multi_matrix(matrix1, matrix2):
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    return matrix1.dot(matrix2)

# Example 3
m1 = np.array([[0, 1, 2], [2,-3, 1]])
m2 = np.array([[1,-3],[6, 1], [0,-1]])
result = matrix_multi_matrix(m1, m2)
print(result) # Output: [[6 -1] [-16 -10]]
