import numpy as np

class OneLayerNeuralNet:

    def __init__(self, S, K):
        self.W1 = -0.1 + 0.2 * np.random.rand(S, K)

    def calculate(self, X, beta):
        U = np.transpose(self.W1) @ X
        Y = 1 / (1 + np.exp(-beta * U))
        return Y

    #proces uczenia sieci
    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        liczbaPrzykladow = input_matrix.shape[1]
        for i in range(0, number_of_epochs):
            nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
            X = np.matrix(input_matrix[:,int(nrPrzykladu)])
            X = X.T 
            Y = self.calculate(X, beta)
            D = output_matrix[:,int(nrPrzykladu)] - Y
            E = D* beta * Y.T * (1 - Y)
            dW = learning_factor * X * E.T
            self.W1 = self.W1 + dW