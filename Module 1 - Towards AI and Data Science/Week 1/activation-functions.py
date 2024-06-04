import math

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

def relu(x):
    return max(0, x)

def elu(x, alpha=0.01):
    return x if x > 0 else alpha * (math.e**(x) - 1)

def exercise2():
    print('Input x = ', end='')
    x = input()

    if is_number(x):
        x = float(x)
    else:
        print('x must be a number')
        exit()

    valid_activation_functions = ['sigmoid', 'relu', 'elu']
    print('Input activation function (sigmoid|relu|elu): ', end='')
    activation_function = input()

    if activation_function not in valid_activation_functions:
        print(f'{activation_function} is not supported')
        exit()

    if activation_function == 'sigmoid':
        print(f'sigmoid: f({x}) = {sigmoid(x)}')
    elif activation_function == 'relu':
        print(f'relu: f({x}) = {relu(x)}')
    elif activation_function == 'elu':
        print(f'elu: f({x}) = {elu(x)}')

exercise2()