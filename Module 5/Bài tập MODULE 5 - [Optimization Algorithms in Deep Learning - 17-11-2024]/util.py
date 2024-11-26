import numpy as np

# f(w1, w2) = 0.1(w1)^2 + 2(w2)^2 (1)
def df_w(W):
    w1, w2 = W
    return np.array([0.2*w1, 4*w2])