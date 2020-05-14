from utils import *
from pprint import pprint
import operator
from random import sample
import copy
# only to make input possible
import pandas as pd


class SNN(object):
    """"After all specific network architecture for example
     SNN(input_size, output_size, 4,2) mans two hidden layers first size 4 second size 2"""

    def __init__(self, input_size, output_size, *args):
        # basic parameters
        self.activation = "sigmoid"
        self.batch = 10
        self.learningRate = 0.05
        self.epoch = 5

        # idealization of structure
        self.hidden = list(args)
        self.output = []
        self.weigth = []
        self.bias = []
        self.feedfowardInput = []
        self.accual = []

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
        # metric
        self.accuracy = 0

    # mozna zrobic implentacje relu ale nie ma co
    def fowardPropagate(self, input):
        if self.activation == "sigmoid":
            state = mult_matr(input, self.weigth[0])
            state = sigmoid(state)
            self.accual.append(state)
            for i in range(1, len(self.weigth) - 1):
                state = mult_matr(state, self.weigth[i])
                state = add(state, self.bias[i])
                state = sigmoid(state)
                self.accual.append(state)
            self.output = sigmoid(mult_matr(state, self.weigth[-1]))
        elif self.activation == "relu":
            state = mult_matr(input, self.weigth[0])
            state = relu(state)
            self.accual.append(state)
            for i in range(1, len(self.weigth) - 1):
                state = mult_matr(state, self.weigth[i])
                state = add(state, self.bias[i])
                state = relu(state)
                self.accual.append(state)
            # output sigmoid activated
            self.output = sigmoid(mult_matr(state, self.weigth[-1]))
        else:
            raise ValueError("No known activation function ")

    def backPropagate(self, target):
        self.calulateDelta(target)
        self.gradient.append(mult_matr(self.delta, transpoze(self.weigth[-1])))
        # propagasja blędu
        for i, j in zip(range(len(self.weigth) - 2, -1, -1), range(len(self.weigth) - 1)):
            if i == 0:
                break
            # Gradient from previes layer times trazpoze matrix times derivative of ativation function
            gradRaw = mult_matr(self.gradient[j], transpoze(self.weigth[i]))
            self.gradient.append(
                list(map(operator.mul, gradientCal(self.accual[i], self.activation), gradRaw)))
        # odwrocic liste bo ide po bakpropagcji
        self.gradient = self.gradient[::-1]
        self.gradient.append(self.delta)
        # dla batcha
        if not self.batchGradient:
            for k in self.gradient:
                self.batchGradient.append(k)

    def updateWeigths(self):
        for i in range(len(self.weigth)):
            self.weigth[i] = updateMatrix(self.weigth[i], scalarMult(self.gradient[i], self.learningRate))

    def updateBias(self):
        for i in range(len(self.bias)):
            self.bias[i] = add(self.bias[i], scalarMult(self.gradient[i], self.learningRate))

    # calculate delta on output


    def calulateDelta(self, target):
        error = list(map(operator.sub, target, self.output))
        gradient = gradientCal(self.output, "sigmoid")
        self.delta = list(map(operator.mul, gradient, error))


    def fit(self, train, target):
        for epoch in range(self.epoch):
            for i, j in zip(sample(range(len(train)), len(train)), range(1, len(train) + 1)):
                self.fowardPropagate(train[i])

                self.backPropagate(target[i])
                if j % self.batch == 0:
                    for k in range(len(self.gradient)):
                        self.gradient[k] = scalarMult(self.batchGradient[k], 1 / self.batch)
                    self.updateBias()
                    self.updateWeigths()
                    self.batchGradient = []

                else:
                    if self.batchGradient:
                        self.batchGradient = add(self.batchGradient, self.gradient)
                # reset gradients
                self.gradient = []
                self.delta = []
                self.accual = []

    def predict(self, test, target):
        outputvec = []
        correct = 0
        for i in range(len(test)):
            self.fowardPropagate(test[i])
            out = hardPrediction(self.output)
            outputvec.append(out)
            if out[0] == target[i][0]:
                correct += 1
        self.accuracy = float(correct) / len(test)
        return outputvec


if __name__ == "__main__":
    # data prep
    data = pd.read_csv("bank_clean.csv")
    big_train, big_test_onehot = prep_data(data)

    # define architecture
    s = SNN(2049, 2, 100, 100)
    s.activation = "relu"
    # fit model
    s.fit(big_train[50:100], big_test_onehot[50:100])
    # make prediction
    s.predict(big_train[:5], big_test_onehot[:5])
    print(s.accuracy)
