import numpy as np
from sklearn.neural_network import MLPClassifier


class SkLearnNeuralNet:
    def __init__(self,
                 number_of_inputs,
                 number_of_outputs,
                 neurons_in_hidden_layer = 0,
                 bias = False):

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.neurons_in_hidden_layer = neurons_in_hidden_layer
        self.bias = bias

        print("Trying to create neural net with parameters:")
        self.display_parameters()

        if((neurons_in_hidden_layer == 0) and (bias == False)):
            self.net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(number_of_inputs,))
        elif((neurons_in_hidden_layer == 0) and (bias == True)):
            self.net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(number_of_inputs,))
            #self.net = OneLayerBiasNeuralNet(number_of_inputs, number_of_outputs)
        elif((neurons_in_hidden_layer > 0) and (bias == False)):
            self.net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(number_of_inputs,neurons_in_hidden_layer))
            #self.net = TwoLayersNeuralNet(number_of_inputs, neurons_in_hidden_layer, number_of_outputs)
        elif((neurons_in_hidden_layer > 0) and (bias == True)):
            self.net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(number_of_inputs,neurons_in_hidden_layer))
            #self.net = TwoLayersBiasNeuralNet(number_of_inputs, neurons_in_hidden_layer, number_of_outputs)
        else:
            print("Problem with initiating network!")

    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        self.beta = beta
        X = [[0., 0.], [1., 1.]]
        # y = [0, 1]
        # print(X)
        # print(y)
        input_matrix = input_matrix.transpose()
        output_matrix = output_matrix.transpose()

        # print(input_matrix.shape)
        # number_of_samples = input_matrix.shape[0]
        # new_input = np.array(number_of_samples, input_matrix.shape[1])
        # for i in range(number_of_samples):
        #     print(input_matrix[i,:])
        #     new_input[i] = input_matrix[i,:]

        # print(input_matrix.tolist())
        # print(output_matrix.tolist())

        self.net.fit(input_matrix.tolist(), output_matrix.tolist())

    def calculate(self, input_vectors):
        input_vectors = input_vectors.transpose()
        list_of_results = []
        for input_vector in input_vectors:
            list_of_results.append(self.net.predict(input_vector.reshape(1, -1))[0])
        print(list_of_results)
        return np.array(list_of_results)

    def display_parameters(self):
        print("Number of inputs: ", self.number_of_inputs)
        print("Number of outputs: ", self.number_of_outputs)
        if(self.neurons_in_hidden_layer > 0):
            print("Number of neurons in hidden layer: ", self.neurons_in_hidden_layer)
        print("Bias: ", self.bias)