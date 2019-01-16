import numpy as np
from neural_net.neural_net import *
from neural_net.sklearn_neural_net import *
from data_loader import load_array_from_data
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report
from evaluation_scores import EvaluationScores
from sklearn.model_selection import train_test_split, RepeatedKFold


def compute_metrics(accuracy_type, ideal, real):
    accs = accuracy_score(ideal, real)
    hl = hamming_loss(ideal, real)
    f1s = f1_score(ideal, real, average='macro')
    # print("    ", accuracy_type, ":")
    # print("         accuracy score: ", accs)
    # print("         loss: ", hl)
    # print("         F1 score: ", f1s)
    # print("         classification report: ")
    # print(classification_report(ideal, real))
    return accs, hl, f1s

def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_train_mlp_pred = classifier.predict(X_train)
    y_test_mlp_pred = classifier.predict(X_test)
    accs_train, hl_train, f1s_train = compute_metrics("Train", y_train, y_train_mlp_pred)
    accs_test, hl_test, f1s_test = compute_metrics("Test", y_test, y_test_mlp_pred)
    return EvaluationScores(accs_train, hl_train, f1s_train, accs_test, hl_test, f1s_test)

def make_experiment(neural_net, X, y):
    N_FOLDS = 2
    N_REPEATS = 5
    cv5x2 = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=36851234)
    feature_selection_NN = EvaluationScores()
    for train_index, test_index in cv5x2.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        feature_selection_NN = feature_selection_NN + evaluate_classifier(neural_net, X_train, X_test, y_train, y_test)
    feature_selection_NN = feature_selection_NN / (N_REPEATS * N_FOLDS)

    feature_selection_NN.display_results("MLP")


X,y = load_array_from_data("data/cancer")


T_NEURONS_IN_HIDDEN_LAYER = [3,10,30,60,90]
for number_of_neurons in T_NEURONS_IN_HIDDEN_LAYER:
    print("ILOŚĆ NEURONÓW ", number_of_neurons)
    neural_net = MLPClassifier(hidden_layer_sizes=(number_of_neurons,), activation='logistic', solver='sgd', max_iter=20000, learning_rate_init=0.1, learning_rate='constant')
    make_experiment(neural_net, X, y)

T_LEARNING_FACTORS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.99]
for learning_rate in T_LEARNING_FACTORS:
    print("WSPÓŁCZYNNIK UCZENIA ", learning_rate)
    neural_net = MLPClassifier(hidden_layer_sizes=(len(X[0])), activation='logistic', solver='sgd', max_iter=20000, learning_rate_init=learning_rate, learning_rate='constant')
    make_experiment(neural_net, X, y)
    neural_net = MLPClassifier(hidden_layer_sizes=(len(X[0]), 10), activation='logistic', solver='sgd', max_iter=20000, learning_rate_init=learning_rate, learning_rate='constant')
    make_experiment(neural_net, X, y)

#T_NUMBER_OF_EPOCHS = [100, 200, 300, 400, 500, 700, 900, 1000, 2000, 5000, 10000, 20000, 40000]
T_NUMBER_OF_EPOCHS = [100]
for epochs in T_NUMBER_OF_EPOCHS:
    print("ILOŚĆ EPOK ", epochs)
    neural_net = MLPClassifier(hidden_layer_sizes=(len(X[0])), activation='logistic', solver='sgd', max_iter=epochs, learning_rate_init=0.1, learning_rate='constant')
    make_experiment(neural_net, X, y)
    neural_net = MLPClassifier(hidden_layer_sizes=(len(X[0]), 10), activation='logistic', solver='sgd', max_iter=epochs, learning_rate_init=0.1, learning_rate='constant')
    make_experiment(neural_net, X, y)
