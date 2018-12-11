


import numpy as np

class TwoLayersBiasNeuralNet:

    def __init__(self, S, K1, K2):
        self.W1 = -0.1 + 0.2 * np.random.rand(S+1, K1)
        self.W2 = -0.1 + 0.2 * np.random.rand(K1+1, K2)

    def calculate(self, input_vector, beta):
        neural_net_result = np.zeros((2,input_vector.shape[1]))
        for j in range(0,input_vector.shape[1]):
            Y2po = self.calculate_two_layers_net_with_bias_output(np.asmatrix(input_vector[:,int(j)]).T, beta)[1]
            neural_net_result[0][j] = Y2po[0][0]
            neural_net_result[1][j] = Y2po[1][0]
        return neural_net_result

    def calculate_two_layers_net_with_bias_output(self, X, beta):
        X1 = np.append(X,[[1]],0)
        U1 = np.transpose(self.W1) @ X1
        Y1 = 1 / (1 + np.exp(-beta * U1))
        X2=np.append(Y1,[[1]],0)
        U2 = np.transpose(self.W2) @ X2
        Y2 = 1 / (1 + np.exp(-beta * U2))
        return Y1, Y2

    #proces uczenia sieci
    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        numberOfExamples = input_matrix.shape[1]
        for i in range(0,number_of_epochs):
            exampleNumber = np.ceil(np.random.rand() * numberOfExamples-1)
            X = np.matrix(input_matrix[:,int(exampleNumber)])
            X1 = np.append(X,[[1]],1)
            X = X.T 
            Y1,Y2 = self.calculate_two_layers_net_with_bias_output(X, beta)
            X2 = np.append(Y1,[[1]],0)
            D2 = output_matrix[:,int(exampleNumber)] - Y2
            E2 = D2* beta * Y2.T * (1 - Y2)
            D1 = self.W2[:-1,:] * E2
            E1 =  D1* beta * Y1.T * (1 - Y1)
            dW1 = learning_factor * X1.T * E1.T
            dW2 = learning_factor * X2 * E2.T
            self.W1 = self.W1 + dW1
            self.W2 = self.W2 + dW2
