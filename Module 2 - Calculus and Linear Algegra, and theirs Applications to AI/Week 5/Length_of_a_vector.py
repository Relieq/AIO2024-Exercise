import numpy as np

def compute_vector_length(v):
    v = np.array(v)
    return np.sqrt(np.sum(v**2))

# Example
vector = np.array([-2, 4, 9, 21])
result = compute_vector_length([vector])
print(round(result, 2)) # Output: 23.28