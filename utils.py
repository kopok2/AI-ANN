from random import random


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

    result = [[sum([x[i][k] * y[k][j] for k in K]) for j in J] for i in I]

    if len(result) == 1:
        return [val for sublist in result for val in sublist]

    return result


def random_initializer(rows, columns):
    return base_initializer(rows, columns, lambda: random.randint(0, 1))


def zero_initializer(rows, columns=1):
    return base_initializer(rows, columns, lambda: -1)


def base_initializer(rows, columns, generator):
    return [[generator()] * columns] * rows
