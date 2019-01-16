import numpy as np
from neural_net.neural_net import *
from neural_net.sklearn_neural_net import *
from data_loader import load_array_from_data
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report


def compute_metrics(accuracy_type, ideal, real):
    accs = accuracy_score(ideal, real)
    hl = hamming_loss(ideal, real)
    f1s = f1_score(ideal, real, average='macro')
    print("    ", accuracy_type, ":")
    print("         accuracy score: ", accs)
    print("         loss: ", hl)
    print("         F1 score: ", f1s)
    print("         classification report: ")
    print(classification_report(ideal, real))
    return accs, hl, f1s

X,y = load_array_from_data("data/cancer")

neural_net = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd')
neural_net.fit(X,y)
y_predicted = neural_net.predict(X)
compute_metrics("MLP", y, y_predicted)

