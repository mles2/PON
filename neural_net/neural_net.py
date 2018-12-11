import numpy as np
from neural_net.one_layer_net import OneLayerNeuralNet
from neural_net.one_layer_bias_net import OneLayerBiasNeuralNet
from neural_net.two_layers_net import TwoLayersNeuralNet
from neural_net.two_layers_bias_net import TwoLayersBiasNeuralNet

class NeuralNet:
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
            self.net = OneLayerNeuralNet(number_of_inputs, number_of_outputs)
        elif((neurons_in_hidden_layer == 0) and (bias == True)):
            self.net = OneLayerBiasNeuralNet(number_of_inputs, number_of_outputs)
        elif((neurons_in_hidden_layer > 0) and (bias == False)):
            self.net = TwoLayersNeuralNet(number_of_inputs, neurons_in_hidden_layer, number_of_outputs)
        elif((neurons_in_hidden_layer > 0) and (bias == True)):
            self.net = TwoLayersBiasNeuralNet(number_of_inputs, neurons_in_hidden_layer, number_of_outputs)
        else:
            print("Problem with initiating network!")

    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        self.beta = beta
        self.net.learn(input_matrix, output_matrix, number_of_epochs, beta, learning_factor)

    def calculate(self, input_vector):
        return self.net.calculate(input_vector, self.beta)

    def display_parameters(self):
        print("Number of inputs: ", self.number_of_inputs)
        print("Number of outputs: ", self.number_of_outputs)
        if(self.neurons_in_hidden_layer > 0):
            print("Number of neurons in hidden layer: ", self.neurons_in_hidden_layer)
        print("Bias: ", self.bias)
