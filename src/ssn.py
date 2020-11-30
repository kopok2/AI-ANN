from utils import *
from pprint import pprint
import operator
from random import sample
import copy
import re
import csv
import ast
# only to make prediction possible
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
class SNN(object):
    """"After all specific network architecture for example
     SNN(input_size, output_size, 4,2) mans two hidden layers first size 4 second size 2"""

    def __init__(self, input_size, output_size, *args):
        # basic parameters
        self.activation = "relu"
        self.outputactivation = "softmax"
        self.batch = 30
        self.learningRate = 0.5
        self.epoch = 5
        self.momentum = 0.8
        # idealization of structure
        self.hidden = list(args)
        self.output = []
        self.weigth = []
        self.bias = []
        self.feedfowardInput = []
        self.accual = []
        self.confmatrix = []
        self.outputvector = []
        self.outputvectorTrain = []
        self.gradientmoment = []
        # init of weights
        self.weigth.append(base_initializer(input_size, self.hidden[0], random))
        for i in range(len(self.hidden) - 1):
            self.weigth.append(base_initializer(self.hidden[i], self.hidden[i + 1], random))
        for i in range(len(self.hidden)):
            self.bias.append(vector_initializer(self.hidden[i], random))
        self.weigth.append(base_initializer(self.hidden[-1], output_size, random))

        self.outputbias = vector_initializer(output_size, random)
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
            if self.outputactivation == "softmax":
                self.output = softmax(add(mult_matr(state, self.weigth[-1]), self.outputbias))
                self.outputvectorTrain.append(self.output)
            elif self.outputactivation == "sigmoid":
                self.output = sigmoid(add(mult_matr(state, self.weigth[-1]), self.outputbias))
                self.outputvectorTrain.append(self.output)
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
            if self.outputactivation == "softmax":
                self.output = softmax(add(mult_matr(state, self.weigth[-1]), self.outputbias))
                self.outputvectorTrain.append(self.output)
            elif self.outputactivation == "sigmoid":
                self.output = sigmoid(add(mult_matr(state, self.weigth[-1]), self.outputbias))
                self.outputvectorTrain.append(self.output)
        else:
            raise ValueError("No known activation function ")

    def backPropagate(self, target):
        self.calulateDelta(target)
        self.gradient.append(mult_matr(self.delta, transpoze(self.weigth[-1])))
        # propagasja blędu
        for i, j in zip(range(len(self.weigth) - 2, -1, -1), range(len(self.weigth) - 1)):
            if i == 0: break
            # Gradient from previes layer times trazpoze matrix times derivative of ativation function
            gradRaw = mult_matr(self.gradient[j], transpoze(self.weigth[i]))
            self.gradient.append(
                list(map(operator.mul, gradientCal(self.accual[i], self.activation), gradRaw)))
        # odwrocic liste
        self.gradient = self.gradient[::-1]
        self.gradient.append(self.delta)
        # dla batcha
        if not self.batchGradient:
            for k in self.gradient:
                self.batchGradient.append(k)


    def updateWeigths(self):
        for i in range(len(self.weigth)):
            self.weigth[i] = updateMatrix(self.weigth[i], add(
                scalarMult(self.gradient[i], self.learningRate),
                scalarMult(self.gradientmoment[i], self.momentum)))

    def updateBias(self):
        self.outputbias = add(self.outputbias, add(
            scalarMult(self.gradient[-1], self.learningRate),
            scalarMult(self.gradientmoment[-1], self.momentum)))
        for i in range(len(self.bias)):
            self.bias[i] = add(self.bias[i], add(
                scalarMult(self.gradient[i], self.learningRate),
                scalarMult(self.gradientmoment[i], self.momentum)))
    # calculate delta on output


    def calulateDelta(self, target):
        error = list(map(operator.sub, target, self.output))
        gradient = gradientCal(self.output, "sigmoid")
        self.delta = list(map(operator.mul, gradient, error))


    def fit(self, train, target):
        for epoch in range(self.epoch):
            start = time.time()
            for i, j in zip(sample(range(len(train)), len(train)), range(1, len(train) + 1)):
                self.fowardPropagate(train[i])
                self.backPropagate(target[i])
                if j % self.batch == 0:
                    if not self.gradientmoment: self.gradientmoment = copy.deepcopy(self.batchGradient)
                    for k in range(len(self.gradient)):
                        self.gradient[k] = self.batchGradient[k]
                    self.updateBias()
                    self.updateWeigths()
                    self.gradientmoment = copy.deepcopy(self.batchGradient)
                    self.batchGradient = []
                else:
                    if self.batchGradient:
                        self.batchGradient = add(self.batchGradient, self.gradient)
                # reset gradients
                self.gradient = []
                self.delta = []
                self.accual = []

            print(self.outputvectorTrain)
            self.outputvectorTrain = []
            print(f"Zakonczone epoke numer {epoch} po czasie {time.time() - start}")

    def predict(self, test, target):
        for i in range(len(test)):
            self.fowardPropagate(test[i])
            out = hardPrediction(self.output)
            self.outputvector.append(out)
        self.confmatrix = create_conf_matrix(decode(target), decode(self.outputvector), 2)
        self.accuracy = calc_accuracy(self.confmatrix)
        print("CONFUSION MATRIX\n")
        pprint(self.confmatrix)
        print(f"accuracy = {self.accuracy}")
        return self.outputvector

    def load_weight(self, file_name):
        super().__init__()
        # parsing name and seting structure
        name = file_name.split(sep=",")
        args = []
        for arg in name:
            args.append(re.sub('[^A-Za-z0-9]+', '', arg))
        # setting architecure
        arch = list(map(int, args[:-2]))
        # add binary output

        arch.insert(1, 2)
        # print(arch)
        self.__init__(*arch)
        self.outputactivation = args[-1]
        self.activation = args[-2]
        # archiceture done now load the weigths
        self.weigth = []
        self.bias = []
        with open(f"wgh/{file_name}.csv") as f:
            csv_file = csv.reader(f, delimiter=",")
            j = 0
            for i, row in enumerate(csv_file, 0):
                # print(row)
                if i % 2 == 0:
                    self.weigth.append(list(map(ast.literal_eval, list(row))))
                else:
                    self.bias.append(list(map(ast.literal_eval, list(row))))
                if i % 2 == 1:
                    j += 1
            self.outputbias = self.bias[-1]
            self.bias = self.bias[:-1]

    def save_weight(self):

        name = f"[{len(self.weigth[0])},{[len(w) for w in self.weigth[1:]]}]"
        print(name)
        ativation = (self.activation, self.outputactivation)

        self.bias.append(self.outputbias)
        with open(f"wgh/{name, *ativation}.csv", "w", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerows(self.weigth)
            writer.writerows(self.bias)


def load_dataset(path="bank.csv", verbose=False):
    print("Loading data...")
    dataset = pd.read_csv(path, dtype=str)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def split_dataset(dataset):
    print("Spliting dataset...")
    X = dataset.drop(['deposit', 'age', 'balance'], axis=1)
    y = dataset['deposit']
    z = dataset[['age', 'balance']]
    return X, y, z

if __name__ == "__main__":
    df = load_dataset(verbose=False)

    X, y, z = split_dataset(df)
    z = z.replace(" ", "0.0", regex=True)
    z = z.apply(pd.to_numeric)

    X = pd.DataFrame(OneHotEncoder().fit_transform(X).toarray())
    y = pd.DataFrame(OneHotEncoder().fit_transform(np.array(y).reshape(-1, 1)).toarray()).loc[:, 1]
    X = pd.concat([X, z], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    X = prep_test_data_x(x_test)
    y = prep_test_data_y(y_test)
    #define architecture
    s = SNN(10, 10, 5)

    s.load_weight("('[2047,[1000, 1000, 10]]', 'sigmoid', 'sigmoid')")
    s.save_weight()

# a = s.predict(X,y)

#print(a)
    # fit model
