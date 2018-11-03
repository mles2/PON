import numpy as np
import xlsxwriter
from neural_net import *
from data_loader import load_data

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('parametrySieci.xlsx')
worksheet = workbook.add_worksheet()

Wpo=np.matrix
W2po=np.matrix

actual_row = 0
actual_column = 0

#output xlsx column settings
ONE_LAYER_RESULT_COLUMN = 0
ONE_LAYER_BIAS_RESULT_COLUMN = 1
TWO_LAYERS_RESULT_COLUMN = 2
TWO_LAYERS_RESULT_BIAS_COLUMN = 3
BETA_COLUMN = 4
LEARNING_FACTOR_COLUMN = 5
HIDDEN_LAYER_COLUMN = 6
EPOCHS_COLUMN = 7


T_BETA = [2,4,7,11,20]
T_LEARNING_FACTORS = [0.01,0.1,0.2, 0.3, 0.5]
T_NEURONS_IN_HIDDEN_LAYER = [3,10,30,60,90]
T_NUMBER_OF_EPOCHS = [100, 1000, 10000, 50000, 80000]
CROSSVALIDATION_PARAMETER = 3

MEANING_PARAMETER = 5

#atr - ilość atrubutów
#count - ilość klas rozpoznawania
#P - macierz atrybutów
#T - macierz diagnoz
#teach - epoki uczenia
#wykonanie kodu zwiazanego z siecia
def make_experiment_with_one_layer(atr, count, P, T, teach, beta, learning_factor):
    sr = 0
    for s in range (0, MEANING_PARAMETER):
        neural_net = NeuralNet(atr, count)
        sk = 0
        for i in range(0,CROSSVALIDATION_PARAMETER):
            aUcz, aTest, dUcz, dTest = create_training_and_testing_data(P,T,i,int(P.shape[1]/CROSSVALIDATION_PARAMETER))
            neural_net.learn(aUcz, dUcz, teach, beta, learning_factor)
            neural_net_result = neural_net.calculate(aTest)
            sk += accuracy(dTest,neural_net_result) 
        sr += sk/CROSSVALIDATION_PARAMETER
        print("Network accuracy: ", sk/CROSSVALIDATION_PARAMETER)
    worksheet.write(actual_row, actual_column, sr/MEANING_PARAMETER)
    print("Mean accuracy: ", sr/MEANING_PARAMETER)

def make_experiment_with_one_layer_with_bias(atr, count, P, T, teach, beta, learning_factor):
    sr = 0
    for s in range (0, MEANING_PARAMETER):
        neural_net = NeuralNet(atr,count, 0, True)
        sk = 0
        for i in range(0,CROSSVALIDATION_PARAMETER):
            aUcz, aTest, dUcz, dTest = create_training_and_testing_data(P,T,i,int(P.shape[1]/CROSSVALIDATION_PARAMETER))
            neural_net.learn(aUcz, dUcz, teach, beta, learning_factor)
            neural_net_result = np.zeros((2,aTest.shape[1]))
            for j in range(0,aTest.shape[1]):
                Y2po = neural_net.calculate(np.asmatrix(aTest[:,int(j)]).T)
                neural_net_result[0][j] = Y2po[0][0]
                neural_net_result[1][j] = Y2po[1][0]
            sk += accuracy(dTest,np.asmatrix(neural_net_result)) 
        sr += sk/CROSSVALIDATION_PARAMETER
        print("Network accuracy: ", sk/CROSSVALIDATION_PARAMETER)

    worksheet.write(actual_row, actual_column, sr/MEANING_PARAMETER)
    print("Mean accuracy: ", sr/MEANING_PARAMETER)


def make_experiment_with_two_layers(atr, count, P, T, teach, hidden, beta, learning_factor):
    sr = 0
    for s in range (0, MEANING_PARAMETER):
        neural_net = NeuralNet(atr, count, hidden)
        sk = 0
        for i in range(0,CROSSVALIDATION_PARAMETER):
            aUcz, aTest, dUcz, dTest = create_training_and_testing_data(P,T,i,int(P.shape[1]/CROSSVALIDATION_PARAMETER))
            neural_net.learn(aUcz, dUcz, teach, beta, learning_factor)
            neural_net_result = neural_net.calculate(aTest)[1]
            sk += accuracy(dTest,neural_net_result) 
        sr += sk/CROSSVALIDATION_PARAMETER
        print("Network accuracy: ", sk/CROSSVALIDATION_PARAMETER)
    worksheet.write(actual_row, actual_column, sr/MEANING_PARAMETER)
    print("Mean accuracy: ", sr/MEANING_PARAMETER)

def make_experiment_with_two_layers_with_bias(atr, count, P, T, teach, hidden, beta, learning_factor):
    sr = 0
    for s in range (0, MEANING_PARAMETER):
        neural_net = NeuralNet(atr,count, hidden, True)
        sk = 0
        for i in range(0,CROSSVALIDATION_PARAMETER):
            aUcz, aTest, dUcz, dTest = create_training_and_testing_data(P,T,i,int(P.shape[1]/CROSSVALIDATION_PARAMETER))
            neural_net.learn(aUcz, dUcz, teach, beta, learning_factor)
            neural_net_result = np.zeros((2,aTest.shape[1]))
            for j in range(0,aTest.shape[1]):
                Y2po = neural_net.calculate(np.asmatrix(aTest[:,int(j)]).T)[1]
                neural_net_result[0][j] = Y2po[0][0]
                neural_net_result[1][j] = Y2po[1][0]
            sk += accuracy(dTest,np.asmatrix(neural_net_result)) 
        sr += sk/CROSSVALIDATION_PARAMETER
        print("Network accuracy: ", sk/CROSSVALIDATION_PARAMETER)
    worksheet.write(actual_row, actual_column, sr/MEANING_PARAMETER)
    print(sr/MEANING_PARAMETER)

#T - porprawne odpowiedzi
#Ypo - uzyskane odpowiedzi
#obliczenie skuteczności sieci
def accuracy(T,Ypo):
    poprawne = 0
    ilosc = Ypo.shape[1]
    for i in range(0, ilosc): 
        t1=T[0,i]
        t2=T[1,i]
        y1=Ypo[0,i]
        y2=Ypo[1,i]
        if(abs(t1-y1) < 0.4 and abs(t2-y2) < 0.4):
            poprawne += 1
    return (poprawne/ilosc)*100

#macierz atrybutów, macierz diagnoz, numer iteracji, wielkosc przedziału out
def create_training_and_testing_data(atr_array,diag_array, iter, zakres):
    poczatekOut = int(iter*zakres)
    koniecOut = int((iter+1)*zakres)
    dlugosc = atr_array.shape[1]

    aTest = atr_array[: ,poczatekOut:koniecOut]
    aUcz = np.concatenate((atr_array[: ,0:poczatekOut], atr_array[: ,koniecOut:dlugosc]),axis=1)   

    dTest = diag_array[: ,poczatekOut:koniecOut]
    dUcz = np.concatenate((diag_array[: ,0:poczatekOut], diag_array[: ,koniecOut:dlugosc]),axis=1)

    return aUcz, aTest, dUcz, dTest

############################################################################
#glowna funkcja

data_array, atr_array = load_data('data/cancer')

diagnoses = data_array[:,-1]
diag_array = []

for d in diagnoses:
    if d == 2:
        diag_array += [[1,0]]
    elif d == 4:
        diag_array += [[0,1]]

diag_array = np.asmatrix(diag_array).transpose()

nowy_pacjent = ([[0.9],[0.5],[0.5],[0.3],[0.6],[0.7],[0.7],[0.9],[0.1]])

worksheet.write(actual_row, ONE_LAYER_RESULT_COLUMN, '1 warstwa')
worksheet.write(actual_row, ONE_LAYER_BIAS_RESULT_COLUMN, '1 warstwa +b')
worksheet.write(actual_row, TWO_LAYERS_RESULT_COLUMN, '2 warstwy')
worksheet.write(actual_row, TWO_LAYERS_RESULT_BIAS_COLUMN, '2 warstwy +b')
worksheet.write(actual_row, BETA_COLUMN, 'Beta')
worksheet.write(actual_row, LEARNING_FACTOR_COLUMN, 'WspUcz')
worksheet.write(actual_row, HIDDEN_LAYER_COLUMN, 'W Ukryta')
worksheet.write(actual_row, EPOCHS_COLUMN, 'Epoki uczenia')
actual_row+=1

#####################################################################################################
# Wpływ ilosci epok uczenia 

T_BETA = [2]
T_LEARNING_FACTORS = [0.99]
T_NEURONS_IN_HIDDEN_LAYER = [10]
T_NUMBER_OF_EPOCHS = [20000]

for t_epokiuczenia in T_NUMBER_OF_EPOCHS:
    for t_beta in T_BETA:
        for t_wspucz in T_LEARNING_FACTORS:
            worksheet.write(actual_row, BETA_COLUMN, t_beta)
            worksheet.write(actual_row, LEARNING_FACTOR_COLUMN, t_wspucz)
            worksheet.write(actual_row, EPOCHS_COLUMN, t_epokiuczenia)

            actual_column=ONE_LAYER_RESULT_COLUMN

            make_experiment_with_one_layer(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_beta, t_wspucz)
            #nowa_diagnoza = calculate_one_layer_net_output(Wpo,nowy_pacjent, beta)
            actual_column+=1

            make_experiment_with_one_layer_with_bias(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_beta, t_wspucz)
            #nowa_diagnoza = calculate_one_layer_net_with_bias_output(Wpo,nowy_pacjent, beta)
            actual_column+=1

            for t_wu in T_NEURONS_IN_HIDDEN_LAYER:
                worksheet.write(actual_row, HIDDEN_LAYER_COLUMN, t_wu)
                worksheet.write(actual_row, BETA_COLUMN, t_beta)
                worksheet.write(actual_row, LEARNING_FACTOR_COLUMN, t_wspucz)
                worksheet.write(actual_row, EPOCHS_COLUMN, t_epokiuczenia)
                make_experiment_with_two_layers(atr_array.shape[0], diag_array.shape[0], atr_array,diag_array, t_epokiuczenia, t_wu, t_beta, t_wspucz)
            # a, nowa_diagnoza = calculate_two_layers_net_output(Wpo, W2po, nowy_pacjent, beta)
                actual_column+=1

                make_experiment_with_two_layers_with_bias(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_wu, t_beta, t_wspucz)
            # a, nowa_diagnoza = calculate_two_layers_net_with_bias_output(Wpo, W2po, nowy_pacjent, beta)

                actual_row+=1
                actual_column=TWO_LAYERS_RESULT_COLUMN

            actual_column=ONE_LAYER_RESULT_COLUMN

workbook.close()

