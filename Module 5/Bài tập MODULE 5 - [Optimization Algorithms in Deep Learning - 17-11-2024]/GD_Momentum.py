from util import *

# f(w1, w2) = 0.1(w1)^2 + 2(w2)^2 (1)
def sgd_momentum(W, dW, lr, V, beta):
    V = beta * V + (1 - beta) * dW
    W = W - lr * V
    return W, V

def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2], dtype=np.float32)
    V = np.zeros_like(W)
    result = [W]
    for i in range(epochs):
        dW = df_w(W)
        W, V = optimizer(W, dW, lr, V, beta=0.5)
        result.append(W)
    return result

if __name__ == '__main__':
    epochs = 30
    lr = 0.6
    result_sgd_momentum = train_p1(sgd_momentum, lr, epochs)
    for i, W in enumerate(result_sgd_momentum):
        print(f'Epoch {i}: {W}')
