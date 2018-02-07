"""
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
"""
from numpy import *
import builtins


def loadDataSet(filename):      # general function to parse tab -delimited floats
    dataMat = []                # assume last column is target value
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataset, feature, value):
    """
    在给定特征和特征值的情况下，该函数通过数组过滤的方式将上述数据集合切分得到两个子集并返回

    :param dataset: 数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return:
    """
    mat0 = []
    mat1 = []

    try:
        # 判断是否是最大值，如果是，则返回空值
        if value < max(dataset[:, feature]):
            mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :][0]
        if value >= min(dataset[:, feature]):
            mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :][0]
    except Exception as result:
        print(result)
    finally:
        return mat0, mat1


def regLeaf(dataset):  # returns the value used for each leaf
    return mean(dataset[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def linearSolve(dataSet):   # helper function used in two places
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th position
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]; tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    if builtins.len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on and the value used for that split


def createTree(dataset, leaf_type=regLeaf, error_type=regErr, ops=(1, 4)):
    """
    树构建函数

    :param dataset:
    :param leaf_type:
    :param error_type:
    :param ops:
    :return:
    """
    # assume dataSet is NumPy Mat so we can array filtering
    feat, value = chooseBestSplit(dataset, leaf_type, error_type, ops)  # choose the best split
    if feat is None:
        return value  # if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = value
    lSet, rSet = binSplitDataSet(dataset, feat, value)
    retTree['left'] = createTree(lSet, leaf_type, error_type, ops)
    retTree['right'] = createTree(rSet, leaf_type, error_type, ops)
    return retTree  


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, test_data):
    if shape(test_data)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree
    if isTree(tree['right']) or isTree(tree['left']):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(test_data, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(test_data, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) +\
            sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(test_data[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1]=inDat
    return float(X*model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, test_data, modelEval=regTreeEval):
    m= builtins.len(test_data)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(test_data[i]), modelEval)
    return yHat