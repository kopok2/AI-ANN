from utils import *
from pprint import pprint
import operator

# only to make input posible
import pandas as pd
from sklearn.model_selection import train_test_split

class SNN(object):
    """"After all specifie network architecure for example
     SNN(input_siez,output_size, 4,2) mens two hidden leyers first size 4 second size 2"""

    # to bedzie troche do wyjebania wladuje sie to fit
    def __init__(self, input_size, output_size, *args):
        # basic parameters
        self.activation = "sigmoid"
        self.batch = 1
        self.learningRate = 0.05
        self.epoch = 100

        # incjalization of structure
        self.hidden = list(args)
        self.output = []
        self.weigth = []
        self.bias = []
        self.feedfowardInput = []

        # init of weights
        self.weigth.append(base_initializer(input_size, self.hidden[0], random))
        for i in range(len(self.hidden) - 1):
            self.weigth.append(base_initializer(self.hidden[i], self.hidden[i + 1], random))
        for i in range(len(self.hidden)):
            self.bias.append(vector_initializer(self.hidden[i], random))
        self.weigth.append(base_initializer(self.hidden[-1], output_size, random))

        # back prop inti
        self.delta = []
        self.gradient = []

    """for debug only """
    def displayNetwork(self):
        # print(len(self.train))
        # print(*self.hidden, sep='\n')
        # print(len(self.target))
        pprint(self.gradient)

    def fowardPropagate(self, input):
        if self.activation == "sigmoid":
            state = mult_matr(input, self.weigth[0])
            state = sigmoid(state)
            for i in range(1, len(self.weigth) - 1):
                state = mult_matr(state, self.weigth[i])
                state = add(state, self.bias[i])
                state = sigmoid(state)
            self.output = sigmoid(mult_matr(state, self.weigth[-1]))
        else:
            raise ValueError("No known activation function ")

    def backPropagate(self, target):
        # jesli delta nie jest policzona policz

        self.calulateDelta(target)
        self.gradient.append(mult_matr(self.delta, transpoze(self.weigth[-1])))

        # maly mlyn ale dziala
        for i, j in zip(range(len(self.weigth) - 2, -1, -1), range(len(self.weigth) - 1)):
            if i == 0: break
            self.gradient.append(mult_matr(self.gradient[j], transpoze(self.weigth[i])))
        # raczej nie potrzebne ale dla wikeszj czytelnosci to zrobie
        # divide all by number of weigth sumed?
        self.gradient = self.gradient[::-1]
        self.gradient.append(self.delta)

    def updateWeigths(self):
        for i in range(len(self.weigth)):
            self.weigth[i] = updateMatrix(self.weigth[i], scalarMult(self.gradient[i], self.learningRate))

    def updateBias(self):
        for i in range(len(self.bias)):
            self.bias[i] = add(self.bias[i], scalarMult(self.gradient[i], self.learningRate))

    # pretty static but very egnouh
    def calulateDelta(self, target):
        error = list(map(operator.sub, self.output, target))
        gradient = gradientCal(self.output, self.activation)
        self.delta = list(map(operator.mul, gradient, error))
        self.delta = list(map(lambda x: -x, self.delta))
        #pprint(self.delta)

    def fit(self, train, target):
        for epoch in range(self.epoch):
            for i in range(len(train)):
                self.fowardPropagate(train[i])
                self.backPropagate(target[i])
                self.updateWeigths()
                self.updateBias()
                # reset gradients
                self.gradient = []
                self.delta = []
                pprint(self.output)

            print(f"Epoka numer {epoch}")
            #pprint(self.gradient)



if __name__ == "__main__":
    a = [[0, 0, 0],
         [1, 2, 3],
         [3, 3, 3]]

    c = [13, 100, 300]

    # pprint(transpoze(a))
    # pprint(a[0][1])
    # print(len(a))
    # pprint(updateMatrix(a,c))

    data = pd.read_csv("bank_clean.csv")

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
    big_train = data.values.tolist()
    train = [1, 0, 0, 1]
    test = [1, 1, 1, 1]

    s = SNN(2049, 2, 1000, 1000)

    s.fit(big_train, big_test_onehot)
