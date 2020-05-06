from utils import *


class SNN(object):
    """"After all arguments specifie network architecure for example SNN(arguent, 4,2) mens two hidden leyers first size 4 second size 2"""

    def __init__(self, train, target, activation="sigmoid", batch=1, *args):
        self.train = train
        self.target = target
        self.activation = activation
        self.batch = batch
        self.hidden = list(args)
        # incjalization of structure
        self.weigth = []
        # input to first hidden
        self.weigth.append(base_initializer(len(train), self.hidden[0], random))
        self.weigth.append(
            [base_initializer(self.hidden[i], self.hidden[i + 1], random) for i in range(len(self.hidden) - 1)])
        self.weigth.append(base_initializer(self.hidden[-1], len(self.target), random))

    """for debug only """

    def displayNetwork(self):
        # print(len(self.train))
        # print(*self.hidden, sep='\n')
        # print(len(self.target))
        for i in self.weigth:
            print(mult_matr(self.target, self.weigth[0]))

    def fowardPropagate(self):
        mult_matr(

        )


import numpy as np

if __name__ == "__main__":
    a = [1, 0, 0, 1]
    b = [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]

    # siec = SNN(a ,b ,"sigmoid", 1,2,3)
    # siec.displayNetwork()
    # c = mult_matr(a,b)
    # print(c)
    print(np.matmul(a, b))
