'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt

'''
    定义文本框和箭头格式
'''
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  #判断结点
leafNode     = dict(boxstyle="round4", fc="0.8")    #叶子结点
arrow_args   = dict(arrowstyle="<-")                 #箭头

def getNumLeafs(myTree):
    '''
        获取叶子结点数目
    :param myTree:
    :return: numLeafs
    '''
    numLeafs = 0
    '''
        Python3改变了dict.keys返回的结果，现在返回的是dict_key对象
        支持iterable，但不支持indexable，可以将其明确地转化成list
    '''
    # firstStr = myTree.keys()[0]
    myTreeList = list(myTree.keys())
    firstStr   = myTreeList[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #测试节点的数据类型是否为字典，如果不是字典，那就是叶子结点
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    # firstStr = myTree.keys()[0]
    myTreeList = list(myTree.keys())
    firstStr   = myTreeList[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #测试节点的数据类型是否为字典，如果不是字典，那就是叶子结点
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
        使用文本注解绘制结点
    :param nodeTxt: 结点描述
    :param centerPt: 中心结点
    :param parentPt: 父结点
    :param nodeType: 结点类型
    :return:
    '''
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    '''
        绘制在中间要显示的文本
    :param cntrPt:
    :param parentPt: 父结点位置
    :param txtString: 要显示的字符串
    :return:
    '''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    '''

        绘制决策树
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    '''
    #这个决定了树的x的宽度
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    print('叶子结点数:',numLeafs)
    #depth = getTreeDepth(myTree)
    #firstStr = myTree.keys()[0]  # the text label for this node should be this
    #firstStr = myTree.keys()
    #这个是结点的文本标签
    myTreeList = list(myTree.keys())
    firstStr   = myTreeList[0]
    cntrPt     = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    '''
        创建图像
    :param inTree:
    :return:
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()   #clf():清除当前图像
    #axprops = dict(xticks=[], yticks=[])
    #createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks 无钩号
    #subplot():返回一个用给定表格位置的子图像

    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 仅用于demo的钩号
    #createPlot.ax1的输出结果为:AxesSubplot(0.125,0.11;0.775x0.77)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

    # createPlot(thisTree)