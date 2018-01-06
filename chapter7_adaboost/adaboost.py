"""
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
"""

from numpy import *
import builtins
import os


def load_simple_data():
    """
    生成测试数据

    :return:
    """
    dat_mat = matrix([
        [1.,  2.1],
        [2.,  1.1],
        [1.3,  1.],
        [1.,  1.],
        [2.,  1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels


def loadDataSet(filename):      # general function to parse tab -delimited floats
    """
    读取数据集的通用方法，这个方法能够自动检测出特征的数目，同时，该方法假设最后一个特征是类别标签

    :param filename:
    :return:
    """
    # 获取文件中的列数，其中，除最后一列外，其余的都是特征
    num_feat = builtins.len(open(filename).readline().split('\t'))  # get number of fields
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr =[]
        cur_line = line.strip().split('\t')
        for i in range(num_feat-1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stumpClassify(data_matrix, dimension, thresh_val, threshIneq):  # just classify the data
    """
    通过阈值比较对数据进行分类

    :param data_matrix:
    :param dimension: 维度
    :param thresh_val:
    :param threshIneq:
    :return:
    """
    retArray = ones((shape(data_matrix)[0], 1))   # 创建一个n行1列的矩阵
    if threshIneq == 'lt':
        retArray[data_matrix[:, dimension] <= thresh_val] = -1.0
    else:
        retArray[data_matrix[:, dimension] > thresh_val] = -1.0
    return retArray
    

def buildStump(dataArr, classLabels, D):
    """
    构造单层决策树

    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T  # 将类别标签进行转置
    m, n = shape(dataMatrix)
    numSteps = 10.0     # 用于在所有可能值上进行遍历
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity 将最小错误率设置为正无穷， 之后用于寻找可能的最小错误率
    for i in range(n):  # loop over all dimensions 遍历数据集的所有特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                # call stump classify with i, j, lessThan
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  # calc total error multiplied by D
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                # % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    基于单层决策树的AdaBoost训练过程

    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    # D是一个概率分布向量，因此其所有的元素之和为1.0。为了满足此要求，一开始的所有元素都会被初始化成1/m.
    D = mat(ones((m, 1))/m)   # init D to all equal
    # aggClassEst是一个列向量，记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        # print("D:",D.T)
        # max(error, 1e-16)用于确保在没有错误时不会发生 除零溢出
        # calc alpha, throw in max(error,eps) to account for error=0
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  # store Stump Params in Array
        # print("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)  # exponent for D calc, getting messy
        # 计算新的权重向量D
        D = multiply(D,exp(expon))                              # Calc New D for next iteration
        D = D/D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        # print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数

    :param datToClass:
    :param classifierArr:
    :return:
    """
    dataMatrix = mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(builtins.len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[0][i]['dim'],
                                 classifierArr[0][i]['thresh'],
                                 classifierArr[0][i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[0][i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    """
    ROC(Receiver Operating Characteristic)接收者操作特征，用于衡量分类中的非均衡性的工具

    AUC(Area Under the Curve)曲线下的面积

    :param predStrengths:
    :param classLabels:
    :return:
    """
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor     保留绘制光标的位置
    ySum = 0.0  # variable to calculate AUC       用于计算AUC的值
    numPosClas = sum(array(classLabels) == 1.0)       # 通过数组过滤方式计算正例数目
    yStep = 1/float(numPosClas)
    xStep = 1/float(builtins.len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    # 构建画笔
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 在所有排序值上进行循环
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX, cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    # 判断图像文件是否存在，如果不存在，则保存
    if not os.path.exists("adaboost.png"):
        plt.savefig("adaboost.png")
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)
