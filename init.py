import numpy as np
import xlsxwriter
from neural_net import *
from data_loader import readcsv

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('parametrySieci.xlsx')
worksheet = workbook.add_worksheet()

Wpo=np.matrix
W2po=np.matrix

row = 0
col = 0
colBeta = 4
colWSpUcz = 5
colWU = 6
colEpoki = 7
c1=0
c1b=1
c2=2
c2b=3

T_BETA = [2,4,7,11,20]
T_WSPUCZ = [0.01,0.1,0.2, 0.3, 0.5]
T_WARSTWAUKRYTA = [3,10,30,60,90]
T_EPOKIUCZENIA = [100, 1000, 10000, 50000, 80000]
KROSWALIDACJA = 3

USREDNIENIE = 5

#atr - ilość atrubutów
#count - ilość klas rozpoznawania
#P - macierz atrybutów
#T - macierz diagnoz
#teach - epoki uczenia
#wykonanie kodu zwiazanego z siecia
def wykonaj(atr, count, P, T, teach, beta, WspUcz):
    sr = 0
    for s in range (0, USREDNIENIE):
        Wprzed = init(atr, count)
        sk = 0
        for i in range(0,KROSWALIDACJA):
            aUcz, aTest, dUcz, dTest = dwaCiagi(P,T,i,int(P.shape[1]/KROSWALIDACJA))
            Wpo = ucz(Wprzed, aUcz, dUcz, teach, beta, WspUcz)
            Ypo = dzialaj(Wpo, aTest, beta)
            sk += skutecznosc(dTest,Ypo) 
        sr += sk/KROSWALIDACJA
    worksheet.write(row, col, sr/USREDNIENIE)
    print(sr/USREDNIENIE)
    return Wpo



def wykonajBias(atr, count, P, T, teach, beta, WspUcz):
    sr = 0
    for s in range (0, USREDNIENIE):
        Wprzed = initBias(atr, count)
        sk = 0
        for i in range(0,KROSWALIDACJA):
            aUcz, aTest, dUcz, dTest = dwaCiagi(P,T,i,int(P.shape[1]/KROSWALIDACJA))
            Wpo = uczBias(Wprzed, aUcz, dUcz, teach, beta, WspUcz)
            Ypo = np.zeros((2,aTest.shape[1]))
            for j in range(0,aTest.shape[1]):
                Y2po = dzialajBias(Wpo,np.asmatrix(aTest[:,int(j)]).T, beta)
                Ypo[0][j] = Y2po[0][0]
                Ypo[1][j] = Y2po[1][0]
            sk += skutecznosc(dTest,np.asmatrix(Ypo)) 
        sr += sk/KROSWALIDACJA

    worksheet.write(row, col, sr/USREDNIENIE)
    print(sr/USREDNIENIE)
    return Wpo


def wykonaj2(atr, count, P, T, teach, hidden, beta, WspUcz):
    sr = 0
    for s in range (0, USREDNIENIE):
        W1przed,W2przed = init2(atr,hidden,count)
        
        sk = 0
        for i in range(0,KROSWALIDACJA):
            aUcz, aTest, dUcz, dTest = dwaCiagi(P,T,i,int(P.shape[1]/KROSWALIDACJA))
            W1po, W2po = ucz2(W1przed, W2przed, aUcz, dUcz, teach, beta, WspUcz)
            Y2po = dzialaj2(W1po, W2po, aTest, beta)[1]
            sk += skutecznosc(dTest,Y2po) 
        sr += sk/KROSWALIDACJA
    worksheet.write(row, col, sr/USREDNIENIE)
    print(sr/USREDNIENIE)
    return W1po, W2po

def wykonaj2bias(atr, count, P, T, teach, hidden, beta, WspUcz):
    sr = 0
    for s in range (0, USREDNIENIE):
        W1przed,W2przed = init2bias(atr,hidden,count)
        
        sk = 0
        for i in range(0,KROSWALIDACJA):
            aUcz, aTest, dUcz, dTest = dwaCiagi(P,T,i,int(P.shape[1]/KROSWALIDACJA))
            W1po, W2po = ucz2bias(W1przed,W2przed,aUcz, dUcz,teach, beta, WspUcz)
            Ypo = np.zeros((2,aTest.shape[1]))
            for j in range(0,aTest.shape[1]):
                Y2po = dzialaj2bias(W1po, W2po, np.asmatrix(aTest[:,int(j)]).T, beta)[1]
                Ypo[0][j] = Y2po[0][0]
                Ypo[1][j] = Y2po[1][0]
            sk += skutecznosc(dTest,np.asmatrix(Ypo)) 
        sr += sk/KROSWALIDACJA
    worksheet.write(row, col, sr/USREDNIENIE)
    print(sr/USREDNIENIE)
    return W1po, W2po

#T - porprawne odpowiedzi
#Ypo - uzyskane odpowiedzi
#obliczenie skuteczności sieci
def skutecznosc(T,Ypo):
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
def dwaCiagi(atr_array,diag_array, iter, zakres):
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

nowy_pacjent = ([[0.9],[0.5],[0.5],[0.3],[0.6],[0.7],[0.7],[0.9],[0.1]])

worksheet.write(row, c1, '1 warstwa')
worksheet.write(row, c1b, '1 warstwa +b')
worksheet.write(row, c2, '2 warstwy')
worksheet.write(row, c2b, '2 warstwy +b')
worksheet.write(row, colBeta, 'Beta')
worksheet.write(row, colWSpUcz, 'WspUcz')
worksheet.write(row, colWU, 'W Ukryta')
worksheet.write(row, colEpoki, 'Epoki uczenia')
row+=1

#####################################################################################################
# Wpływ ilosci epok uczenia 

T_BETA = [2]
T_WSPUCZ = [0.99]
T_WARSTWAUKRYTA = [10]
T_EPOKIUCZENIA = [20000]

for t_epokiuczenia in T_EPOKIUCZENIA:
    for t_beta in T_BETA:
        for t_wspucz in T_WSPUCZ:
            worksheet.write(row, colBeta, t_beta)
            worksheet.write(row, colWSpUcz, t_wspucz)
            worksheet.write(row, colEpoki, t_epokiuczenia)

            col=c1

            Wpo = wykonaj(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_beta, t_wspucz)
            #nowa_diagnoza = dzialaj(Wpo,nowy_pacjent, beta)
            col+=1

            Wpo = wykonajBias(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_beta, t_wspucz)
            #nowa_diagnoza = dzialajBias(Wpo,nowy_pacjent, beta)
            col+=1

            for t_wu in T_WARSTWAUKRYTA:
                worksheet.write(row, colWU, t_wu)
                worksheet.write(row, colBeta, t_beta)
                worksheet.write(row, colWSpUcz, t_wspucz)
                worksheet.write(row, colEpoki, t_epokiuczenia)
                Wpo, W2po = wykonaj2(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_wu, t_beta, t_wspucz)
            # a, nowa_diagnoza = dzialaj2(Wpo, W2po, nowy_pacjent, beta)
                col+=1

                Wpo, W2po = wykonaj2bias(atr_array.shape[0],diag_array.shape[0],atr_array,diag_array,t_epokiuczenia, t_wu, t_beta, t_wspucz)
            # a, nowa_diagnoza = dzialaj2bias(Wpo, W2po, nowy_pacjent, beta)

                row+=1
                col=c2

            col=c1

workbook.close()