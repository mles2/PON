

class EvaluationScores:

    def __init__(self, accuracy_score_train=0, hamming_loss_train=0, f1_score_train=0, accuracy_score_test=0,
                 hamming_loss_test=0, f1_score_test=0):
        self.accuracy_score_train = accuracy_score_train
        self.hamming_loss_train = hamming_loss_train
        self.f1_score_train = f1_score_train
        self.accuracy_score_test = accuracy_score_test
        self.hamming_loss_test = hamming_loss_test
        self.f1_score_test = f1_score_test

    def __add__(self, other):
        accuracy_score_train = self.accuracy_score_train + other.accuracy_score_train
        hamming_loss_train = self.hamming_loss_train + other.hamming_loss_train
        f1_score_train = self.f1_score_train + other.f1_score_train
        accuracy_score_test = self.accuracy_score_test + other.accuracy_score_test
        hamming_loss_test = self.hamming_loss_test + other.hamming_loss_test
        f1_score_test = self.f1_score_test + other.f1_score_test

        return EvaluationScores(accuracy_score_train, hamming_loss_train, f1_score_train, accuracy_score_test,
                                hamming_loss_test, f1_score_test)

    def __truediv__(self, other):
        accuracy_score_train = self.accuracy_score_train / other
        hamming_loss_train = self.hamming_loss_train / other
        f1_score_train = self.f1_score_train / other
        accuracy_score_test = self.accuracy_score_test / other
        hamming_loss_test = self.hamming_loss_test / other
        f1_score_test = self.f1_score_test / other

        return EvaluationScores(accuracy_score_train, hamming_loss_train, f1_score_train, accuracy_score_test,
                                hamming_loss_test, f1_score_test)

    def display_results(self, title=""):
        #print(title)
        #print("Accuracy score train: ", self.accuracy_score_train)
        #print(" Hamming loss train ", self.hamming_loss_train)
        #print(" F1 score train ", self.f1_score_train)
        print(" Accuracy score test: ", 100*self.accuracy_score_test)
        #print(" Hamming loss test ", self.hamming_loss_test)
        #print(" F1 score test ", self.f1_score_test)
        
