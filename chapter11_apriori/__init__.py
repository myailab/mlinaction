# @Time    : 2017/9/21 23:47
# @Author  : myailab
# @Site    : http://www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm
# !/usr/bin/python
import chapter11_apriori.apriori as ap

if __name__ == "__main__":
    type = 'test_apriori'
    # type = 'mushroom_test'

    if type == 'test_apriori':
        print('Apriori算法测试')
        # dataset = ap.loadDataSet()

        dataset_tmp = ap.loadDataSet2()
        dataset = []
        for d in dataset_tmp:
            #未去重的数据集
            dataset.append(d)
            # 去重后的数据集
            # if d not in dataset:
            #     dataset.append(d)
        # dataset = ap.loadDataSet3()
        print(len(dataset))
        L, suppData = ap.apriori(dataset, minSupport=0.5)
        rules = ap.generateRules(L, suppData, minConf=0.7)
        # print("rules:\n", rules)
        print("关联规则:\n", rules)
        C1 = ap.createC1(dataset)
        D = map(set, dataset)
        L1, suppData0 = ap.scanD(D, C1, 0.5)
        print('L1:\n', L1)
        # print('suppData0:\n', suppData0)
        print('支持项集0:\n', suppData0)
        # knn.handwritingClassTest()
    elif type == 'mushroom_test':
        print('发现毒蘑菇的相似特征')
        mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
        L, suppData = ap.apriori(mushDataSet, minSupport=0.1)
        for item in L[1]:
            # intersection:求集合的交集
            if item.intersection('2'):
                print(item+"\n")
                exit()
    else:
        print('no matching')