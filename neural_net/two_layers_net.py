import numpy as np

class TwoLayersNeuralNet:

    def __init__(self, S, K1, K2):
        self.W1 = -0.1 + 0.2 * np.random.rand(S, K1)
        self.W2 = -0.1 + 0.2 * np.random.rand(K1, K2)

    def calculate_two_layers_net_output(self, X, beta):
        U1 = np.transpose(self.W1) @ X
        Y1 = 1 / (1 + np.exp(-beta * U1))
        U2 = np.transpose(self.W2) @ Y1
        Y2 = 1 / (1 + np.exp(-beta * U2))
        return Y1,Y2

    def calculate(self, X, beta):
        return self.calculate_two_layers_net_output(X, beta)[1]

    #proces uczenia sieci
    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        numberOfExamples = input_matrix.shape[1]
        for i in range(0,number_of_epochs):
            exampleNumber = np.ceil(np.random.rand() * numberOfExamples-1)
            X1 = np.matrix(input_matrix[:,int(exampleNumber)])
            X1 = X1.T 
            Y1,Y2 = self.calculate_two_layers_net_output(X1, beta)
            D2 = output_matrix[:,int(exampleNumber)] - Y2
            E2 = D2* beta * Y2.T * (1 - Y2)
            D1 = self.W2 * E2
            E1 =  D1* beta * Y1.T * (1 - Y1)
            dW1 = learning_factor * X1 * E1.T
            dW2 = learning_factor * Y1 * E2.T
            self.W1 = self.W1 + dW1
            self.W2 = self.W2 + dW2