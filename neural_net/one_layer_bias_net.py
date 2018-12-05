import numpy as np

class OneLayerBiasNeuralNet:
    def __init__(self, S, K):
        self.W1 = -0.1 + 0.2 * np.random.rand(S+1, K)

    def calculate(self, input_vector, beta):
        neural_net_result = np.zeros((2,input_vector.shape[1]))
        for j in range(0,input_vector.shape[1]):
            Y2po = self.calculate_one_layer_net_with_bias_output(np.asmatrix(input_vector[:,int(j)]).T, beta)
            neural_net_result[0][j] = Y2po[0][0]
            neural_net_result[1][j] = Y2po[1][0]

        return neural_net_result

    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        liczbaPrzykladow = input_matrix.shape[1]
        W = self.W1
        for i in range(0, number_of_epochs):
            nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
            X = np.matrix(input_matrix[:,int(nrPrzykladu)])
            X1 = np.append(X,[[1]],1)
            X = X.T 
            Y = self.calculate_one_layer_net_with_bias_output(X, beta)
            D = output_matrix[:,int(nrPrzykladu)] - Y
            E = D* beta * Y.T * (1 - Y)
            dW = learning_factor * X1.T * E.T
            self.W1 = self.W1 + dW

    def calculate_one_layer_net_with_bias_output(self, X, beta):
        X1 = np.append(X,[[1]],0)
        U = np.transpose(self.W1) @ X1
        Y = 1 / (1 + np.exp(-beta * U))
        return Y

