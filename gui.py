from tkinter import *
import numpy as np
import csv

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Cancer Diagnose")

        labelFrame = Frame(master)
        labelFrame.grid(row = 0, column = 0,  )

        entryFrame = Frame(master)
        entryFrame.grid(row = 0, column = 1)

        diagFrame = Frame(master)
        diagFrame.grid(row = 1)

        self.entryText = IntVar

        self.l1 = Label(labelFrame, text="Clump Thickness")
        self.l1.grid(sticky=W,pady=1)

        self.e1 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e1.grid()

        self.l2 = Label(labelFrame, text="Uniformity of Cell Size")
        self.l2.grid(sticky=W,pady=1)

        self.e2 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e2.grid()

        self.l3 = Label(labelFrame, text="Uniformity of Cell Shape")
        self.l3.grid(sticky=W,pady=1)

        self.e3 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e3.grid()

        self.l4 = Label(labelFrame, text="Marginal Adhesion")
        self.l4.grid(sticky=W,pady=1)

        self.e4 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e4.grid()

        self.l5 = Label(labelFrame, text="Single Epithelial Cell Size")
        self.l5.grid(sticky=W,pady=1)

        self.e5 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e5.grid()

        self.l6 = Label(labelFrame, text="Bare Nuclei")
        self.l6.grid(sticky=W,pady=1)

        self.e6 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e6.grid()

        self.l7 = Label(labelFrame, text="Bland Chromatin")
        self.l7.grid(sticky=W,pady=1)
               
        self.e7 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e7.grid()

        self.l8 = Label(labelFrame, text="Normal Nucleoli")
        self.l8.grid(sticky=W,pady=1)

        self.e8 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e8.grid()

        self.l9 = Label(labelFrame, text="Mitoses")
        self.l9.grid(sticky=W,pady=1)

        self.e9 = Entry(entryFrame, textvariable=self.entryText, width = 5)
        self.e9.grid()



        self.greet_button = Button(diagFrame, text="Diagnose", command=self.diagnose)
        self.greet_button.grid(padx=5,pady=5)


        self.lbenign = Label(diagFrame,text = "Benign: ")
        self.lbenign.grid()

        self.lbenignDiag = Label(diagFrame,text = "-")
        self.lbenignDiag.grid(column = 1,row = 1)

        self.lmalignant = Label(diagFrame,text = "Malignant: ")
        self.lmalignant.grid()

        self.lmalignantDiag = Label(diagFrame,text = "-")
        self.lmalignantDiag.grid(column = 1, row = 2)

    

    def diagnose(self):
        nowy_pacjent = ([[int(self.e1.get())/10],[int(self.e2.get())/10],[int(self.e3.get())/10],[int(self.e4.get())/10],[int(self.e5.get())/10],[int(self.e6.get())/10],[int(self.e7.get())/10],[int(self.e8.get())/10],[int(self.e9.get())/10]])
        nowa_diagnoza = dzialaj(Wpo,nowy_pacjent)
        benign = float(nowa_diagnoza[0])*100
        malignant = float(nowa_diagnoza[1])*100
        self.lbenignDiag.config(text='{:.0f} %'.format(benign))
        self.lmalignantDiag.config(text='{:.0f} %'.format(malignant))




def readcsv(filename,dm):
    with open(filename, 'r') as p:
        my_list = [[int(x) for x in rec] for rec in csv.reader(p, delimiter=',')]
    return my_list

Wpo=np.matrix

#inicjacja sieci losowymi wagami
def init(S, K):
    W1 = -0.1 + 0.2 * np.random.rand(S, K)
    return W1

#obliczenie odpowiedzi sieci
def dzialaj(W, X):
    beta = 5
    U = np.transpose(W) @ X
    Y = 1 / (1 + np.exp(-beta * U))
    return Y

#proces uczenia sieci
def ucz(Wprzed, P, T, n):
    liczbaPrzykladow = P.shape[1]
    W = Wprzed
    WspUcz = 0.1
    beta = 5
    for i in range(0, n):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(P[:,int(nrPrzykladu)])
        X = X.T 
        Y = dzialaj(W, X)
        D = T[:,int(nrPrzykladu)] - Y
        E = D* beta * Y.T * (1 - Y)
        dW = WspUcz * X * E.T
        W = W + dW
    Wpo = W
    return Wpo

#wykonanie kodu zwiazanego z siecia
def wykonaj(atr,count,P,T,teach):
    Wprzed = init(atr, count)
    Wpo = ucz(Wprzed, P, T, teach)
    Ypo = dzialaj(Wpo, P)
    skutecznosc(T,Ypo) 
    return Wpo


#obliczenie skuteczności sieci
def skutecznosc(T,Ypo):
    poprawne = 0
    for i in range(0,683):  #TODO: zrobić zeby automatycznie sprawdzało wymiar
        t1=T[0,i]
        t2=T[1,i]
        y1=Ypo[0,i]
        y2=Ypo[1,i]
        if(abs(t1-y1) < 0.4 and abs(t2-y2) < 0.6):
            poprawne += 1
    print("SKUTECZNOŚĆ: ", poprawne/6.83)

#glowna funkcja

data_array = readcsv("./data/cancer",',')
for l in data_array:
    l=np.array(l)

data_array = np.array(data_array)
atr_array = data_array[:,1:-1]
atr_array = atr_array.transpose() / 10

diagnoses = data_array[:,-1]
diag_array = []

for d in diagnoses:
    if d == 2:
        diag_array += [[1,0]]
    elif d == 4:
        diag_array += [[0,1]]

diag_array = np.asmatrix(diag_array).transpose()

Wpo = wykonaj(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,100000)


root = Tk()
my_gui = GUI(root)
root.mainloop()
