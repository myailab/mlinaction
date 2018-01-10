"""
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
"""
from numpy import *
import operator
import builtins
from os import listdir


def classify0(inX, dataset, labels, k):
    """
    k近邻算法

    :param inX:
    :param dataset:
    :param labels:
    :param k:
    :return:
    """
    """
        shape函数是numpy.core.formnumeric中的函数，它的功能是读取矩阵的长度
        shape[0]:读取矩阵第一维的长度
        先返回列，后返回行
    """
    dataSetSize = dataset.shape[0]
    """
        tile(矩阵，(行，列))
        如果只有一维，则是列方向重复
    """
    diffMat = tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2    # 平方
    """
        sum():参数为1：行求和，参数为0：列求和
    """
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # 开平方
    """ 
        argsort():返回数组从小到大的索引值，如果是二维数组
            argsort(x, axis=0):列排序
            argsort(x, axis=1):行排序
            argsort(x):升序
            argsort(-x):降序
    """
    sortedDistIndicies = distances.argsort()
    class_count = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1
    # sorted_class_count = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    """
    将文件转换成矩阵

    :param filename: 文件的路径
    :return:
    """
    fr = open(filename)
    number_of_lines = builtins.len(fr.readlines())  # get the number of lines in the file
    """
        zeros(行，列):返回一个给定形状和类型的用0填充的数组
        zeros(shape, dtype=float, order='c')
        c:行优先
        F:列优先
    """
    return_matrix = zeros((number_of_lines, 3))  # prepare matrix to return
    class_label_vector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector


def autoNorm(dataset):
    """
    归一化数据集

    :param dataset:
    :return:
    """
    min_values = dataset.min(0)
    max_values = dataset.max(0)
    ranges = max_values - min_values
    # normal_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normal_dataset = dataset - tile(min_values, (m, 1))
    normal_dataset = normal_dataset / tile(ranges, (m, 1))  # element wise divide
    return normal_dataset, ranges, min_values


def datingClassTest(file_path):
    hoRatio = 0.50  # hold out 10%
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data set from file
    dating_data_matrix, dating_labels = file2matrix(file_path)  # load data set from file
    normMat, ranges, min_values = autoNorm(dating_data_matrix)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], dating_labels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, dating_labels[i]))
        if classifierResult != dating_labels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    """
    将图片转换成向量

    :param filename:
    :return:
    """
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector


def handwritingClassTest():
    """
    手写识别系统->测试数据

    :return:
    """
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = builtins.len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = builtins.len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))