from numpy import *
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
       # lineArr.append(1)
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
        xMat = mat(xArr); yMat = mat(yArr).T
        xTx = xMat.T*xMat
        if linalg.det(xTx) == 0.0:
            print ("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T*yMat)
        return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat



def gradient_descent(xArr, yArr, iterations, alpha):
    xmat = mat(xArr); ymat = mat(yArr).T
    m = shape(ymat)[0]
    n = shape(xmat)[1]
    ws = []
    for i in range(n):
        w = [np.random.rand()]
        ws.append(w)
    previous_costs = []
    previous_ws = [ws]
    for i in range(iterations):
        prediction = (xmat * ws)
        error = prediction - ymat
        cost = 1/(2*m) * (error.T * error)
        previous_costs.append(cost)
        ws = ws - (alpha * (1/m) * (xmat.T * error))
        previous_ws.append(ws)
                
    return previous_ws, previous_costs
        

