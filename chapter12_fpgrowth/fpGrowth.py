"""
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter

FP-growth(Frequent Pattern -growth)

"""


class treeNode:
    def __init__(self, name_value, num_occurrence, parent_node):
        """

        :param name_value: 节点的名字
        :param num_occurrence: 出现的次数
        :param parent_node: 当前结点的父结点
        """
        self.name = name_value
        self.count = num_occurrence
        self.nodeLink = None  # 用于链接相似的元素项
        self.parent = parent_node      # needs to be updated
        self.children = {} 
    
    def inc(self, num_occurrence):
        """
        对count变量增加给定值
        :param num_occurrence:
        :return:
        """
        self.count += num_occurrence
        
    def display(self, ind=1):
        """
        此方法用于将树以文本形式显示
        :param ind:
        :return:
        """
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(ind + 1)


def createTree(dataset, min_support=1):  # create FP-tree from dataset but don't mine
    """
    使用数据集以及最小支持度作为参数来构建FP树
    :param dataset:
    :param min_support:
    :return:
    """
    headerTable = {}  # 头指针表，这是一个字典
    # go over dataSet twice
    # 遍历数据集两次
    for trans in dataset:  # first pass counts frequency of occurance
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataset[trans]
    # for k in headerTable.keys():  # remove items not meeting minSup
    for k in list(headerTable.keys()):  # remove items not meeting minSup
        if headerTable[k] < min_support:  # 移除不满足最小支持度的元素项
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    # print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0:  # 如果没有元素项满足要求，则退出
        return None, None  # if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # reformat headerTable to use Node link
    # print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None)  # create tree
    for tranSet, count in dataset.items():  # go through dataset 2nd time
        localD = {}
        for item in tranSet:  # put transaction items in order
            # 根据全局频率对每个事务中的元素进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # 使用排序后的频率项集对树进行填充
            updateTree(orderedItems, retTree, headerTable, count)  # populate tree with ordered freq item set
    return retTree, headerTable  # return tree and header table


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:  # check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count)  # increment count
    else:   # add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:  # update header table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        # 不断迭代调用自身，每次调用时会去掉列表中的第一个元素
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):   # this version does not use recursion
    # 确保节点链接指向树中该元素项的每一个实例
    while nodeToTest.nodeLink is not None:    # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    # 从叶子结点上溯到根结点
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    # 发现前缀路径
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 递归查找频繁项集
    # 从头指针表的底端开始
    # 原来的p[1]是节点类型，无法进行比较，所以需要对p[1]进行取值，所以修改为如下代码：p[1][0]
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # (sort header table)
    for basePat in bigL:  # start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print 'finalFrequent Item: ',newFreqSet    # append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print 'condPattBases :',basePat, condPattBases
        # 2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        # print 'head from conditional tree: ', myHead
        if myHead is not None:  # 3. mine cond. FP-tree
            print('conditional tree for: ',newFreqSet)
            myCondTree.display(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def load_simple_data():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    # 实现从列表到字典的转换
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


import twitter
from time import sleep
import re


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

# minSup = 3
# simpDat = loadSimpDat()
# initSet = createInitSet(simpDat)
# myFPtree, myHeaderTab = createTree(initSet, minSup)
# myFPtree.disp()
# myFreqList = []
# mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
