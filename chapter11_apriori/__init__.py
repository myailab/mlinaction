# @Time    : 2017/9/21 23:47
# @Author  : myailab
# @Site    : http://www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm
#!/usr/bin/python
import chapter11_apriori.apriori as ap

if __name__ == "__main__":
    type = 'testApriori'

    if ( type == 'testApriori' ):
        print('Apriori算法测试')
        dataset = ap.loadDataSet()
        L, suppData = ap.apriori(dataset, minSupport=0.5)
        rules = ap.generateRules(L, suppData, minConf=0.7)
        print(rules)
        exit()
        C1 = ap.createC1(dataset)
        D  = map(set, dataset)
        L1, suppData0 = ap.scanD(D, C1, 0.5)
        print('L1:', L1)
        print('suppData0:', suppData0)
        # knn.handwritingClassTest()
    elif ( type == 'datingTest' ):
        print('apriori')
        # filePath = "testdata\\datingTestSet2.txt"
        # knn.datingClassTest(filePath)
    else :
        print ('no matching')