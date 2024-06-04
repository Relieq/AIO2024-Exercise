def is_integer(x):
    try:
        if x == int(x):
            return True
        else:
            return False
    except ValueError:
        return False

def recall(tp, fn):
    return tp / (tp + fn)

def precision(tp, fp):
    return tp / (tp + fp)

def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r)

def exercise1(tp, fp, fn):
    if not is_integer(tp):
        print('tp must be int')
        return False
    if not is_integer(fp):
        print('fp must be int')
        return False
    if not is_integer(fn):
        print('fn must be int')
        return False
    if tp <= 0 or fp <= 0 or fn <= 0:
        print('tp and fp and fn must be greater than zero')
        return False
    print(f'precision is {precision(tp, fp)}')
    print(f'recall is {recall(tp, fn)}')
    print(f'f1-score is {f1_score(tp, fp, fn)}')

# Example:
exercise1(2, 3, 4)
print('---------------------')
exercise1('a', 3, 4)
print('---------------------')
exercise1(2, 'a', 4)
print('---------------------')
exercise1(2, 3, 'a')
print('---------------------')
exercise1(2, 3, 0)
print('---------------------')
exercise1(2.1, 3, 0)