from random import random
import math
import operator


def mult_matr(a, b):
    x = a
    y = b
    if not isinstance(a[0], list):
        x = [x]
    if not isinstance(b[0], list):
        y = [y]

    I = range(len(x))
    J = range(len(y[0]))
    K = range(len(x[0]))
    if len(x[0]) != len(y):
        raise ValueError(f"Dim x = {len(x[0])} is not equal to {len(y)}")

    result = [[sum([x[i][k] * y[k][j] for k in K]) for j in J] for i in I]

    if len(result) == 1:
        return [val for sublist in result for val in sublist]

    return result


def random_initializer(rows, columns):
    return base_initializer(rows, columns, lambda: random.randint(0, 1))


def zero_initializer(rows, columns=1):
    return base_initializer(rows, columns, lambda: -1)


def base_initializer(rows, columns, generator):
    return [[generator() for i in range(columns)] for j in range(rows)]


def vector_initializer(size, generator):
    return [generator() for i in range(size)]


def sigmoid(vec):
    return list(map(lambda x: 1 - 1 / (1 + math.exp(x)) if x < 0 else 1 / (1 + math.exp(-x)), vec))


def relu(vec):
    return list(map(lambda x: max(0, x), vec))


def sigmoidDiriv(vec):
    sig = lambda x: (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))
    return list(map(sig, vec))


def reluDiriv(vec):
    return list(map(lambda x: 1 if x > 0 else 0, vec))


def gradientCal(layer, activation):
    if activation == "sigmoid":
        gradient = sigmoidDiriv(layer)
    elif activation == "relu":
        gradient = reluDiriv(layer)
    else:
        raise ValueError("Not known activation function ")
    return gradient


def transpoze(matrix):
    return list(map(list, zip(*matrix)))


def add(vec1, vec2):
    return list(map(operator.add, vec1, vec2))


def updateMatrix(matrix, value):
    matrix = transpoze(matrix)
    return transpoze([list(map(lambda x: x + value[i], matrix[i]))
                      for i in range(len(matrix))])


def scalarMult(vec, scalar):
    return list(map(lambda x: x * scalar, vec))


def hardPrediction(vec):
    return list(map(round, vec))


def prep_data(data):
    data.drop(columns=["Unnamed: 0"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    big_test = data["1.1"].values.tolist()
    big_test_onehot = []

    for i in big_test:
        if i:
            big_test_onehot.append([1, 0])
        else:
            big_test_onehot.append([0, 1])
    data.drop(columns=["1.1"], inplace=True)
    return data.values.tolist(), big_test_onehot


def prep_test_data_y(test_data):
    big_test = test_data.values.tolist()
    return [[1, 0] if i else [0, 1] for i in big_test]


def prep_test_data_x(test_data):
    test_data.reset_index(drop=True, inplace=True)
    return test_data.values.tolist()


def fibonacci(n):
    if n < 0:
        print("Incorrect input")
        # First Fibonacci number is 0
    elif n == 1:
        return 0
    # Second Fibonacci number is 1
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_range(fib_range):
    return (fibonacci(n) for n in fib_range)


def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return transpoze(m)


def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t


def decode(vec):
    return [1 if i[0] == 1 else 0 for i in vec]


def softmax(vec):
    maks = max(vec)
    vec = list(map(lambda x: x - maks, vec))
    sum = math.fsum(list(map(math.exp, vec)))
    return list(map(lambda x: x / sum, list(map(math.exp, vec))))
