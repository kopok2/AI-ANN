from random import random

from functools import reduce
from operator import add


def mult_matr(a, b):
    result = [[0] * len(b[0]) for _ in range(len(a))]
    for row in range(len(b[0])):
        for column in range(len(a)):
            result[row][column] = reduce(add, map(lambda x: x[0] * x[1],
                                                  zip(a[row], (b[x][column] for x in range(len(a[0]))))))
    return result


def random_initializer(rows, columns):
    return base_initializer(rows, columns, lambda: random.randint(0, 1))


def zero_initializer(rows, columns=1):
    return base_initializer(rows, columns, lambda: -1)


def base_initializer(rows, columns, generator):
    vector = []
    for i in range(rows):
        vector[i] = []
        for j in range(columns):
            vector[i][j] = generator()

    return vector
