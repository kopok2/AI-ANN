from utils import *
import operator
from random import sample
# only to make input posible
import pandas as pd

class SNN(object):
    """"After all specifie network architecure for example
     SNN(input_siez,output_size, 4,2) mens two hidden leyers first size 4 second size 2"""
    def __init__(self, input_size, output_size, *args):
        # basic parameters
        self.activation = "sigmoid"
        self.batch = 3
        self.learningRate = 1
        self.epoch = 1

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
        self.batchGradient = []
        # metirc
        self.accuracy = 0

    # mozna zrobic imlentacje relu ale nie ma co
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
        # licz delte
        self.calulateDelta(target)
        self.gradient.append(mult_matr(self.delta, transpoze(self.weigth[-1])))
        #propagasja blędu
        for i, j in zip(range(len(self.weigth) - 2, -1, -1), range(len(self.weigth) - 1)):
            if i == 0: break
            self.gradient.append(mult_matr(self.gradient[j], transpoze(self.weigth[i])))
        #odwrucic liste bo ide po bakpropagcji
        self.gradient = self.gradient[::-1]
        self.gradient.append(self.delta)
        # dla batcha
        if self.batchGradient == []:
            for k in self.gradient:
                self.batchGradient.append(k)

    def updateWeigths(self):
        for i in range(len(self.weigth)):
            self.weigth[i] = updateMatrix(self.weigth[i], scalarMult(self.gradient[i], self.learningRate))

    def updateBias(self):
        for i in range(len(self.bias)):
            self.bias[i] = add(self.bias[i], scalarMult(self.gradient[i], self.learningRate))

    # calucate delta on output
    def calulateDelta(self, target):
        error = list(map(operator.sub, self.output, target))
        gradient = gradientCal(self.output, self.activation)
        self.delta = list(map(operator.mul, gradient, error))
        self.delta = list(map(lambda x: -x, self.delta))

    def fit(self, train, target):
        for epoch in range(self.epoch):
            for i, j in zip(sample(range(len(train)), len(train)), range(1, len(train) + 1)):
                self.fowardPropagate(train[i])
                self.backPropagate(target[i])
                # batch update moznapelej ale nie ma co
                if j % self.batch == 0:
                    for k in range(len(self.gradient)):
                        self.gradient[k] = scalarMult(self.batchGradient[k], 1 / self.batch)
                    self.updateBias()
                    self.updateWeigths()
                    self.batchGradient = []

                else:
                    if self.batchGradient != []:
                        self.batchGradient = add(self.batchGradient, self.gradient)
                # reset gradients
                self.gradient = []
                self.delta = []
                print(f"output = {self.output}, target = {target[i]}")
            print(f"Epoka numer {epoch + 1}")

    def predict(self, test, target):
        corect = 0
        outputvec = []
        for i in range(len(test)):
            self.fowardPropagate(test[i])
            out = hardPrediction(self.output)
            outputvec.append(out)
            if out[0] == target[i][0]:
                corect += 1
            print(f"output = {out}, target = {target[i]}")
        self.accuracy = float(corect) / len(test)
        print(f"accuracy = {self.accuracy}")
        return outputvec

if __name__ == "__main__":
    #data prep
    data = pd.read_csv("bank_clean.csv")
    data.drop(columns=["Unnamed: 0"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    big_test = data["1.1"].values.tolist()
    big_test_onehot = []
    #jak chcesz to zrob to ladniej
    for i in big_test:
        if i:
            big_test_onehot.append([1, 0])
        else:
            big_test_onehot.append([0, 1])
    data.drop(columns=["1.1"], inplace=True)
    big_train = data.values.tolist()

    #defina archiceture
    s = SNN(2049, 2, 1000, 1000)
    # fit model
    s.fit(big_train[50:60], big_test_onehot[50:60])
    # make prediction
    s.predict(big_train[:5], big_test_onehot[:5])
