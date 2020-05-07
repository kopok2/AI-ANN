from utils import *
from pprint import pprint
import operator

# for debug
import numpy as np

class SNN(object):
    """"After all arguments specifie network architecure for example SNN(arguent, 4,2) mens two hidden leyers first size 4 second size 2"""

    def __init__(self, train, target, activation="sigmoid", batch=1, *args):
        self.train = train
        self.target = target
        self.activation = activation
        self.batch = batch
        self.hidden = list(args)
        # incjalization of structure
        self.output = []
        self.weigth = []
        # input to first hidden
        self.weigth.append(base_initializer(len(train), self.hidden[0], random))
        for i in range(len(self.hidden) - 1):
            self.weigth.append(base_initializer(self.hidden[i], self.hidden[i + 1], random))
        self.weigth.append(base_initializer(self.hidden[-1], len(self.target), random))
        # back prop inti
        self.delta = []
    """for debug only """
    def displayNetwork(self):
        # print(len(self.train))
        # print(*self.hidden, sep='\n')
        # print(len(self.target))
        for i in self.weigth:
            pprint(i)

    def fowardPropagate(self):
        if self.activation == "sigmoid":
            state = mult_matr(self.train, self.weigth[0])
            state = sigmoid(state)
            for i in range(1, len(self.weigth) - 1):
                state = mult_matr(state, self.weigth[i])
                state = sigmoid(state)
            self.output = sigmoid(mult_matr(state, self.weigth[-1]))
        else:
            raise ValueError("No known activation function ")

    def backPropagate(self):
        pass

    def calulateDelta(self):
        self.delta = list(map(operator.sub, self.output, self.target))


if __name__ == "__main__":
    a = [1, 0, 0, 1]
    b = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    c = [1, 2, 3, 4.4]
    siec = SNN(a, c, "sigmoid", 1, 2, 3)
    # siec.displayNetwork()
    siec.fowardPropagate()

    siec.calulateDelta()
    print(siec.delta)
    # print(list(map(operator.sub,a,c)))
    # c = base_initializer(4,2,random)
    # print(np.array(c).shape)
    #print(mult_matr(a,b))
