from util import *

# f(w1, w2) = 0.1(w1)^2 + 2(w2)^2 (1)
def RMSprop(W, dW, lr, S, beta, eps):
    S = beta * S + (1 - beta) * dW**2
    W = W - lr * dW / (np.sqrt(S + eps))
    return W, S

def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2], dtype=np.float32)
    S = np.zeros_like(W)
    result = [W]
    for i in range(epochs):
        dW = df_w(W)
        W, S = optimizer(W, dW, lr, S, beta=0.9, eps=1e-6)
        result.append(W)
    return result

if __name__ == '__main__':
    epochs = 30
    lr = 0.3
    result_rmsprop = train_p1(RMSprop, lr, epochs)
    for i, W in enumerate(result_rmsprop):
        print(f'Epoch {i}: {W}')
