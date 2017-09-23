'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


def createDataSet():
    '''
        创建一个测试数据集
    :return:
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
        计算香农熵
    :param dataSet: 原始数据集
    :return:
    '''
    numEntries  = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # the the number of unique elements and their occurance
        #唯一元素的数量和发生次数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #print('分类标签数量：', labelCounts)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
        分割数据集合
    :param dataSet:
    :param axis:
    :param value:
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # print('axis:', axis) : 0
            # print('featVec[axis]:', featVec[axis]) : 0
            # print(featVec) :[1, 1, 'no']
            # print(featVec[:axis]) :[]
            # print(featVec[axis+1:]) :[1, 'no']
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting 删除用于切分的轴
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
        选择最佳特征进行分割
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1  #计算特征的个数，由于最后一列是类别标签，所以去掉
    print('特征个数:',numFeatures)
    #计算基准信息熵
    baseEntropy = calcShannonEnt(dataSet)
    print('基准信息熵为:%.5f' % baseEntropy)
    bestInfoGain = 0.0
    bestFeature  = -1
    # iterate over all the features
    # 遍历所有特征
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        # 创建一个所有同种特征的列表
        # for example in dataSet:
        #     print(example)
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        # 获取集合的唯一值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        print('使用第%s个特征划分数据集得到的信息熵为：%.5f' % (i, newEntropy))
        # calculate the info gain; ie reduction in entropy
        #计算信息增益
        infoGain = baseEntropy - newEntropy
        print('使用第%s个特征划分数据集得到的信息增益为：%.5f' % (i, infoGain))
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
        创建决策树
    :param dataSet: 数据集合
    :param labels: 标签
    :return:
    '''
    # for example in dataSet:
    #     print(example[-1])
    #首先，获取数据集中的分类列表
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # stop splitting when all of the classes are equal
        # 当所有类别都相等时，停止分割
        return classList[0]
    if len(dataSet[0]) == 1:
        # stop splitting when there are no more features in dataSet
        #当数据集中没有特征时，停止分割
        return majorityCnt(classList)
    #选择最佳特征进行分割
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print('最佳特征:', bestFeat)
    bestFeatLabel = labels[bestFeat]
    print('最佳特征的标签:', bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

