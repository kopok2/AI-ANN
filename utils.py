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
