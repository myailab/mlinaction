"""
Created on Jan 8, 2011

@author: Peter
"""
from numpy import *
from time import sleep
import urllib.request
import json
import builtins


def loadDataSet(filename):      # general function to parse tab -delimited floats
    # 获取特征的数量
    features = builtins.len(open(filename).readline().split('\t')) - 1  # get number of fields
    data_matrix = []
    label_matrix = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(features):
            line_arr.append(float(cur_line[i]))
        data_matrix.append(line_arr)
        label_matrix.append(float(cur_line[-1]))
    return data_matrix, label_matrix


def standRegres(x_arr, y_arr):
    """
    标准线性回归，用来计算最佳拟合直线

    :param x_arr:
    :param y_arr:
    :return:
    """
    x_matrix = mat(x_arr)
    y_matrix = mat(y_arr).T
    xTx = x_matrix.T * x_matrix  # 计算x^T.x
    """
    linalg为Numpy提供的一个线性代数库，其中包含很多有用的函数
    """
    if linalg.det(xTx) == 0.0:  # 调用linalg.det()来计算行列式
        # 如果行列式的值为0，则返回错误提示
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_matrix.T * y_matrix)  # .I, 计算矩阵的逆矩阵
    return ws


def lwlr(test_point, x_arr, y_arr, k=1.0):
    """
    局部加权线性回归(Locally Weighted Linear Regression， LWLR)

    :param test_point:
    :param x_arr:
    :param y_arr:
    :param k:
    :return:
    """
    x_matrix = mat(x_arr)
    y_matrix = mat(y_arr).T
    m = shape(x_matrix)[0]
    """
    numpy中的eye()方法，创建一个对角矩阵，参数m为行数
    """
    weights = mat(eye(m))
    for j in range(m):                      # next 2 lines create weights matrix
        diffMat = test_point - x_matrix[j, :]     #
        # 权重大小以指数级衰减，输入参数k控制衰减的速度
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))  # 高斯核对应的权重
    xTx = x_matrix.T * (weights * x_matrix)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_matrix.T * (weights * y_matrix))
    return test_point * ws


def lwlrTest(test_arr, x_arr, y_arr, k=1.0):  # loops over all the data points and applies lwlr to each one
    """
    局部加权线性回归的测试

    :param test_arr:
    :param x_arr:
    :param y_arr:
    :param k:
    :return:
    """
    m = shape(test_arr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return yHat


def lwlrTestPlot(xArr, yArr, k=1.0):  # same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       # easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def ridgeRegres(x_matrix, y_matrix, lam=0.2):
    """
    岭回归(ridge regression)：实现了给定lambda下的岭回归求解

    :param x_matrix:
    :param y_matrix:
    :param lam:
    :return:
    """
    xTx = x_matrix.T * x_matrix
    denom = xTx + eye(shape(x_matrix)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (x_matrix.T * y_matrix)
    return ws


def ridgeTest(x_arr, y_arr):
    xMat = mat(x_arr)
    yMat = mat(y_arr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean     # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)   # calc mean then subtract it off
    xVar = var(xMat, 0)      # calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(x_matrix):  # regularize by columns
    inMat = x_matrix.copy()
    inMeans = mean(inMat, 0)   # calc mean then subtract it off
    inVar = var(inMat, 0)      # calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(x_arr, y_arr, eps=0.01, times=100):
    """
    前向逐步线性回归
    
    :param x_arr: 
    :param y_arr: 
    :param eps: 每次迭代需要调整的步长
    :param times: 迭代次数
    :return: 
    """
    xMat = mat(x_arr)
    yMat = mat(y_arr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean     # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((times, n))  # testing code remove
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(times):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# def scrapePage(inFile, outFile, yr, numPce, origPrc):
#     from bs4 import BeautifulSoup
#     fr = open(inFile)
#     fw = open(outFile,'a') # a is append mode writing
#     soup = BeautifulSoup(fr.read())
#     i=1
#     currentRow = soup.findAll('table', r="%d" % i)
#     while builtins.len(currentRow) != 0:
#         title = currentRow[0].findAll('a')[1].text
#         lwrTitle = title.lower()
#         if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#             newFlag = 1.0
#         else:
#             newFlag = 0.0
#         soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#         if builtins.len(soldUnicde) == 0:
#             print("item #%d did not sell" % i)
#         else:
#             soldPrice = currentRow[0].findAll('td')[4]
#             priceStr = soldPrice.text
#             priceStr = priceStr.replace('$', '')  # strips out $
#             priceStr = priceStr.replace(',', '')  # strips out ,
#             if builtins.len(soldPrice) > 1:
#                 priceStr = priceStr.replace('Free shipping', '')  # strips out Free Shipping
#             print("%s\t%d\t%s" % (priceStr, newFlag, title))
#             fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr, numPce, newFlag, origPrc, priceStr))
#         i += 1
#         currentRow = soup.findAll('table', r="%d" % i)
#     fw.close()


# 预测乐高玩具套装的价格，由于缺乏数据，无法实现
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    # pg = urllib2.urlopen(searchURL)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(builtins.len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = builtins.len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)    # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX-meanTrain)/varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k]=rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1*sum(multiply(meanX, unReg)) + mean(yMat))