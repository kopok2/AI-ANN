from utils import *
from pprint import pprint
import operator

# for debug
import numpy as np

class SNN(object):
    """"After all arguments specifie network architecure for example SNN(arguent, 4,2) mens two hidden leyers first size 4 second size 2"""

    # to bedzie troche do wyjebania wladuje sie to fit
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
        self.learningRate = 0.05
        self.delta = []
        self.gradient = []
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
        # jesli delta nie jest policzona policz
        if len(self.delta) == 0:
            self.calulateDelta()
            self.gradient.append(mult_matr(self.delta, transpoze(self.weigth[-1])))

        # maly mlyn ale dziala
        for i, j in zip(range(len(self.weigth) - 2, -1, -1), range(len(self.weigth) - 1)):
            if i == 0: break
            self.gradient.append(mult_matr(self.gradient[j], transpoze(self.weigth[i])))

        # raczej nie potrzebne ale dla wikeszj czytelnosci to zrobie
        # divide all by number of weigth sumed?
        self.gradient = self.gradient[::-1]
        self.gradient.append(self.delta)

    def batchUpdate(self):
        update = []
        for batch in range(self.batch):
            pass

    def fit(self, train, target):
        pass

    #pretty static but very egnouh
    def calulateDelta(self):
        error = list(map(operator.sub, self.output, self.target))
        gradient = gradientCal(self.output, self.activation)
        self.delta = list(map(operator.mul, gradient, error))

if __name__ == "__main__":
    a = [1, 0, 0, 1]
    b = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    c = [1, 0]
    siec = SNN(a, c, "sigmoid", 1, 2, 3)
    # siec.displayNetwork()
    siec.fowardPropagate()
    # print(len(siec.delta))
    # siec.calulateDelta()
    siec.backPropagate()
    pprint(siec.gradient)
    #print(siec.delta)
    # print(list(map(operator.sub,a,c)))
    # c = base_initializer(4,2,random)
    # print(np.array(c).shape)
    #print(mult_matr(a,b))
