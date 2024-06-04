import random
import math

def loss (optional, y_true, y_pred):
    sum_error = 0
    if optional == 'MAE':
        for i in range(len(y_true)):
            error = abs(y_true[i] - y_pred[i])
            print(f'''loss name: {optional}, sample: {i}, pred: {y_pred[i]}, 
                target: {y_true[i]}, loss: {error}''')
            sum_error += error
        print(f'final {optional}: ', sum_error / len(y_true))
    
    elif optional == 'MSE':
        for i in range(len(y_true)):
            error = (y_true[i] - y_pred[i]) ** 2
            print(f'''loss name: {optional}, sample: {i}, pred: {y_pred[i]}, 
                target: {y_true[i]}, loss: {error}''')
            sum_error += error
        print(f'final {optional}: ', sum_error / len(y_true))
    
    elif optional == 'RMSE':
        for i in range(len(y_true)):
            error = (y_true[i] - y_pred[i]) ** 2
            print(f'''loss name: {optional}, sample: {i}, pred: {y_pred[i]}, 
                target: {y_true[i]}, loss: {error}''')
            sum_error += error
        print(f'final {optional}: ', math.sqrt(sum_error / len(y_true)))
    
    else:
        return None

def exercise3():
    print('Input number of samples (integer number) which are generated: ', end='')
    num_samples = input()
    if not num_samples.isnumeric():
        print('Number of samples must be an integer number')
        exit()
    else:
        num_samples = int(num_samples)
    
    random.seed(0)
    y_true = [random.uniform(0, 10) for i in range(num_samples)]
    y_pred = [random.uniform(0, 10) for i in range(num_samples)]

    print('Input loss name: ', end='')
    optional = input()
    loss(optional, y_true, y_pred)

exercise3()
    

