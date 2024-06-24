import torch
import torch.nn as nn

class MySoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        exps = torch.exp(x)
        return exps / exps.sum()

class SoftmaxStable(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        exps = torch.exp(x - max(x))
        return exps / exps.sum()
# Example:
data = torch.Tensor([5, 2, 4])
my_softmax = MySoftmax()
softmax_stable = SoftmaxStable()
output1 = my_softmax(data)
output2 = softmax_stable(data)
print(f'Output from MySoftmax: {output1}')
print(f'Output from SoftmaxStable: {output2}')