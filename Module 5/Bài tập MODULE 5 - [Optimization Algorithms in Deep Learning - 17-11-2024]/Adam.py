from util import *

# f(w1, w2) = 0.1(w1)^2 + 2(w2)^2 (1)
def Adam(W, dW, lr, V, S, beta1, beta2, t, eps=1e-6):
    V = beta1 * V + (1 - beta1) * dW
    S = beta2 * S + (1 - beta2) * dW**2
    V_corrected = V / (1 - beta1**t)
    S_corrected = S / (1 - beta2**t)
    W = W - lr * V_corrected / (np.sqrt(S_corrected) + eps)
    return W, V, S

def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2], dtype=np.float32)
    V = np.zeros_like(W)
    S = np.zeros_like(W)
    result = [W]
    for i in range(epochs):
        dW = df_w(W)
        W, V, S = optimizer(W, dW, lr, V, S, beta1=0.9, beta2=0.999, t=i+1)
        result.append(W)
    return result

if __name__ == '__main__':
    epochs = 30
    lr = 0.2
    result_adam = train_p1(Adam, lr, epochs)
    for i, W in enumerate(result_adam):
        print(f'Epoch {i}: {W}')
