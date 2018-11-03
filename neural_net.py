import numpy as np

ONE_LAYER = 1
ONE_LAYER_BIAS = 2
TWO_LAYERS = 3
TWO_LAYERS_BIAS = 4

#inicjacja sieci losowymi wagami
def init(S, K):
    W1 = -0.1 + 0.2 * np.random.rand(S, K)
    return W1

def initBias(S,K):
    W1 = -0.1 + 0.2 * np.random.rand(S+1, K)
    return W1

def init2(S,K1,K2):
    W1 = -0.1 + 0.2 * np.random.rand(S, K1)
    W2 = -0.1 + 0.2 * np.random.rand(K1, K2)
    return [W1, W2]

def init2bias(S,K1,K2):
    W1 = -0.1 + 0.2 * np.random.rand(S+1, K1)
    W2 = -0.1 + 0.2 * np.random.rand(K1+1, K2)
    return [W1, W2]

#obliczenie odpowiedzi sieci
def calculate_one_layer_net_output(W, X, beta):
    U = np.transpose(W) @ X
    Y = 1 / (1 + np.exp(-beta * U))
    return Y

def calculate_one_layer_net_with_bias_output(W,X, beta):
    X1 = np.append(X,[[1]],0)
    U = np.transpose(W) @ X1
    Y = 1 / (1 + np.exp(-beta * U))
    return Y

def calculate_two_layers_net_output(W1, W2, X, beta):
    U1 = np.transpose(W1) @ X
    Y1 = 1 / (1 + np.exp(-beta * U1))
    U2 = np.transpose(W2) @ Y1
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2

def calculate_two_layers_net_with_bias_output(W1, W2, X, beta):
    X1 = np.append(X,[[1]],0)
    U1 = np.transpose(W1) @ X1
    Y1 = 1 / (1 + np.exp(-beta * U1))
    X2=np.append(Y1,[[1]],0)
    U2 = np.transpose(W2) @ X2
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2


#proces uczenia sieci
def learn_one_layer_network(weights_matrix, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
    liczbaPrzykladow = input_matrix.shape[1]
    W = weights_matrix
    for i in range(0, number_of_epochs):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(input_matrix[:,int(nrPrzykladu)])
        X = X.T 
        Y = calculate_one_layer_net_output(W, X, beta)
        D = output_matrix[:,int(nrPrzykladu)] - Y
        E = D* beta * Y.T * (1 - Y)
        dW = learning_factor * X * E.T
        W = W + dW
    Wpo = W
    return Wpo

def learn_one_layer_bias_network(weights_matrix, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
    liczbaPrzykladow = input_matrix.shape[1]
    W = weights_matrix
    for i in range(0, number_of_epochs):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(input_matrix[:,int(nrPrzykladu)])
        X1 = np.append(X,[[1]],1)
        X = X.T 
        Y = calculate_one_layer_net_with_bias_output(W, X, beta)
        D = output_matrix[:,int(nrPrzykladu)] - Y
        E = D* beta * Y.T * (1 - Y)
        dW = learning_factor * X1.T * E.T
        W = W + dW
    Wpo = W
    return Wpo

def learn_two_layers_network(first_layer_matrix, second_layer_matrix, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
    liczbaPrzykladow = input_matrix.shape[1]
    W1 = first_layer_matrix
    W2 = second_layer_matrix
    for i in range(0,number_of_epochs):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X1 = np.matrix(input_matrix[:,int(nrPrzykladu)])
        X1 = X1.T 
        Y1,Y2 = calculate_two_layers_net_output(W1, W2, X1, beta)
        D2 = output_matrix[:,int(nrPrzykladu)] - Y2
        E2 = D2* beta * Y2.T * (1 - Y2)
        D1 = W2 * E2
        E1 =  D1* beta * Y1.T * (1 - Y1)
        dW1 = learning_factor * X1 * E1.T
        dW2 = learning_factor * Y1 * E2.T
        W1 = W1 + dW1
        W2 = W2 + dW2
    return W1,W2

def learn_two_layers_bias_network(first_layer_matrix, second_layer_matrix, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
    liczbaPrzykladow = input_matrix.shape[1]
    W1 = first_layer_matrix
    W2 = second_layer_matrix
    for i in range(0,number_of_epochs):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(input_matrix[:,int(nrPrzykladu)])
        X1 = np.append(X,[[1]],1)
        X = X.T 
        Y1,Y2 = calculate_two_layers_net_with_bias_output(W1, W2, X, beta)
        X2 = np.append(Y1,[[1]],0)
        D2 = output_matrix[:,int(nrPrzykladu)] - Y2
        E2 = D2* beta * Y2.T * (1 - Y2)
        D1 = W2[:-1,:] * E2
        E1 =  D1* beta * Y1.T * (1 - Y1)
        dW1 = learning_factor * X1.T * E1.T
        dW2 = learning_factor * X2 * E2.T
        W1 = W1 + dW1
        W2 = W2 + dW2
    return W1,W2

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
            self.weight_matrixes = [init(number_of_inputs, number_of_outputs)]
            self.net_type = ONE_LAYER
        elif((neurons_in_hidden_layer == 0) and (bias == True)):
            self.weight_matrixes = [initBias(number_of_inputs, number_of_outputs)]
            self.net_type = ONE_LAYER_BIAS
        elif((neurons_in_hidden_layer > 0) and (bias == False)):
            self.weight_matrixes = init2(number_of_inputs, neurons_in_hidden_layer, number_of_outputs)
            self.net_type = TWO_LAYERS
        elif((neurons_in_hidden_layer > 0) and (bias == True)):
            self.weight_matrixes = init2bias(number_of_inputs, neurons_in_hidden_layer, number_of_outputs)
            self.net_type = TWO_LAYERS_BIAS
        else:
            print("Problem with initiating network!")

    def learn(self, input_matrix, output_matrix, number_of_epochs, beta, learning_factor):
        self.beta = beta
        if(self.net_type == ONE_LAYER):
            self.weight_matrixes[0] = learn_one_layer_network(self.weight_matrixes[0],input_matrix, output_matrix, number_of_epochs, beta, learning_factor)
        elif(self.net_type == ONE_LAYER_BIAS):
            self.weight_matrixes[0] = learn_one_layer_bias_network(self.weight_matrixes[0], input_matrix, output_matrix, number_of_epochs, beta, learning_factor)
        elif(self.net_type == TWO_LAYERS):
            self.weight_matrixes[0], self.weight_matrixes[1] = learn_two_layers_network(self.weight_matrixes[0], self.weight_matrixes[1], input_matrix, output_matrix, number_of_epochs, beta, learning_factor)
        elif(self.net_type == TWO_LAYERS_BIAS):
            self.weight_matrixes[0], self.weight_matrixes[1] = learn_two_layers_bias_network(self.weight_matrixes[0], self.weight_matrixes[1], input_matrix, output_matrix, number_of_epochs, beta, learning_factor)

    def calculate(self, input_vector):
        if(self.net_type == ONE_LAYER):
            return calculate_one_layer_net_output(self.weight_matrixes[0],input_vector, self.beta)
        elif(self.net_type == ONE_LAYER_BIAS):
            return calculate_one_layer_net_with_bias_output(self.weight_matrixes[0], input_vector, self.beta)
        elif(self.net_type == TWO_LAYERS):
            return calculate_two_layers_net_output(self.weight_matrixes[0], self.weight_matrixes[1], input_vector, self.beta)
        elif(self.net_type == TWO_LAYERS_BIAS):
            return calculate_two_layers_net_with_bias_output(self.weight_matrixes[0], self.weight_matrixes[1], input_vector, self.beta)


    def display_parameters(self):
        print("Number of inputs: ", self.number_of_inputs)
        print("Number of outputs: ", self.number_of_outputs)
        if(self.neurons_in_hidden_layer > 0):
            print("Number of neurons in hidden layer: ", self.neurons_in_hidden_layer)
        print("Bias: ", self.bias)

neural_net_2 = NeuralNet(2,4)
neural_net_1 = NeuralNet(2,3,4)
neural_net_3 = NeuralNet(2,4,0,True)
neural_net_3 = NeuralNet(2,4,2,True)