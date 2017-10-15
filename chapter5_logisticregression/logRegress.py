'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import builtins

def loadDataSet():
    dataMat = []; labelMat = []
    #fr = open('testData/testSet.txt')
    fr = open('testData/testSet2.txt')  #这个是精简后的数据集，只包含5条数据
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    '''
        sigmoid阶跃函数
    '''
    return longfloat(1.0 / (1 + exp(-inX)))


def gradAscent(dataMatIn, classLabels):
    print('dataMatIn before converted:\n', dataMatIn)
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix  转换成Numpy矩阵
    print('dataMatIn after converted:\n', dataMatrix);
    print('classLabels before transposed:\n', classLabels)
    print('classLabels in numpy format:\n', mat(classLabels))
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix  转换成Numpy矩阵
    print('classLabels after transposed:\n', labelMat)

    #获取dataMatrix的维度
    m, n = shape(dataMatrix)
    print("行:%s,列:%s" % (m,n))
    exit()
    alpha = 0.001
    #循环的次数
    maxCycles = 500
    weights = ones((n, 1))
    print('weights:\n', weights)
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)  # matrix mult
        print('h:%s' % (h))
        error = (labelMat - h)  # vector subtraction
        print('labeMat:\n', labelMat)
        print('error:%s' %(error))

        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
        print('weights:%s' %(weights))
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];    ycord1 = []
    xcord2 = [];    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    #???add_subplot()方法的使用方法
    ax = fig.add_subplot(111)
    # ???scatter()方法的使用方法
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');    plt.ylabel('X2');    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, builtins.len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('testData\horseColicTraining.txt');frTest = open('testData\horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10;    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))
