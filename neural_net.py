import numpy as np

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
    return W1, W2

def init2bias(S,K1,K2):
    W1 = -0.1 + 0.2 * np.random.rand(S+1, K1)
    W2 = -0.1 + 0.2 * np.random.rand(K1+1, K2)
    return W1, W2

#obliczenie odpowiedzi sieci
def dzialaj(W, X, beta):
    U = np.transpose(W) @ X
    Y = 1 / (1 + np.exp(-beta * U))
    return Y

def dzialajBias(W,X, beta):
    X1 = np.append(X,[[1]],0)
    U = np.transpose(W) @ X1
    Y = 1 / (1 + np.exp(-beta * U))
    return Y

def dzialaj2(W1, W2, X, beta):
    U1 = np.transpose(W1) @ X
    Y1 = 1 / (1 + np.exp(-beta * U1))
    U2 = np.transpose(W2) @ Y1
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2

def dzialaj2bias(W1, W2, X, beta):
    X1 = np.append(X,[[1]],0)
    U1 = np.transpose(W1) @ X1
    Y1 = 1 / (1 + np.exp(-beta * U1))
    X2=np.append(Y1,[[1]],0)
    U2 = np.transpose(W2) @ X2
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2


#proces uczenia sieci
def ucz(Wprzed, P, T, n, beta, WspUcz):
    liczbaPrzykladow = P.shape[1]
    W = Wprzed
    for i in range(0, n):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(P[:,int(nrPrzykladu)])
        X = X.T 
        Y = dzialaj(W, X, beta)
        D = T[:,int(nrPrzykladu)] - Y
        E = D* beta * Y.T * (1 - Y)
        dW = WspUcz * X * E.T
        W = W + dW
    Wpo = W
    return Wpo

def uczBias(Wprzed, P, T, n, beta, WspUcz):
    liczbaPrzykladow = P.shape[1]
    W = Wprzed
    for i in range(0, n):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(P[:,int(nrPrzykladu)])
        X1 = np.append(X,[[1]],1)
        X = X.T 
        Y = dzialajBias(W, X, beta)
        D = T[:,int(nrPrzykladu)] - Y
        E = D* beta * Y.T * (1 - Y)
        dW = WspUcz * X1.T * E.T
        W = W + dW
    Wpo = W
    return Wpo

def ucz2(W1przed, W2przed, P, T, n, beta, WspUcz):
    liczbaPrzykladow = P.shape[1]
    W1 = W1przed
    W2 = W2przed
    for i in range(0,n):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X1 = np.matrix(P[:,int(nrPrzykladu)])
        X1 = X1.T 
        Y1,Y2 = dzialaj2(W1, W2, X1, beta)
        D2 = T[:,int(nrPrzykladu)] - Y2
        E2 = D2* beta * Y2.T * (1 - Y2)
        D1 = W2 * E2
        E1 =  D1* beta * Y1.T * (1 - Y1)
        dW1 = WspUcz * X1 * E1.T
        dW2 = WspUcz * Y1 * E2.T
        W1 = W1 + dW1
        W2 = W2 + dW2
    return W1,W2

def ucz2bias(W1przed, W2przed, P, T, n, beta, wspUcz):
    liczbaPrzykladow = P.shape[1]
    W1 = W1przed
    W2 = W2przed
    for i in range(0,n):
        nrPrzykladu = np.ceil(np.random.rand() * liczbaPrzykladow-1)
        X = np.matrix(P[:,int(nrPrzykladu)])
        X1 = np.append(X,[[1]],1)
        X = X.T 
        Y1,Y2 = dzialaj2bias(W1, W2, X, beta)
        X2 = np.append(Y1,[[1]],0)
        D2 = T[:,int(nrPrzykladu)] - Y2
        E2 = D2* beta * Y2.T * (1 - Y2)
        D1 = W2[:-1,:] * E2
        E1 =  D1* beta * Y1.T * (1 - Y1)
        dW1 = wspUcz * X1.T * E1.T
        dW2 = wspUcz * X2 * E2.T
        W1 = W1 + dW1
        W2 = W2 + dW2
    return W1,W2
        