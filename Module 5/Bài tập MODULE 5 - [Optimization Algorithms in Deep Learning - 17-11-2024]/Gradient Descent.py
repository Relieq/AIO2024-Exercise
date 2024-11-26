from util import *

# f(w1, w2) = 0.1(w1)^2 + 2(w2)^2 (1)

def sgd(W, dW, lr):
    return W - lr * dW

def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2], dtype=np.float32)
    result = [W]
    for i in range(epochs):
        dW = df_w(W)
        W = optimizer(W, dW, lr)
        result.append(W)
    return result

if __name__ == '__main__':
    epochs = 30
    lr = 0.4
    result_sgd = train_p1(sgd, lr, epochs)
    for i, W in enumerate(result_sgd):
        print(f'Epoch {i}: {W}')
