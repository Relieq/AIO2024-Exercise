def levenshtein_distance(token1, token2):
    rows = len(token1) + 1
    cols = len(token2) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]

    # initialize the distance matrix
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    
    # recursively compute the distance of the two tokens
    for col in range(1, cols):
        for row in range(1, rows):
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + (token1[row-1] != token2[col-1]))      # Cost of substitutions
    
    distances = distance[-1][-1]
    
    return distances

# Example:
print(levenshtein_distance('yu', 'you')) # Output: 1
print(levenshtein_distance('kitten', 'sitting')) # Output: 3