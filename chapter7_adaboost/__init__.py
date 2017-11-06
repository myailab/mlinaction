# @Time    : 2017/9/28 23:07
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm

import chapter7_adaboost.adaboost as ada
from numpy import *

if __name__ == '__main__':
    type = "builds#tump"
    #type = "trainingAdaBoost"
    # type = "hardataset"
    type = 'plotroc'

    dataMat, classLabels = ada.loadSimpData()

    if (type == "buildstump") :
        print("单层决策树生成函数：")
        #D是一个概率分布向量，因此其所有的元素之和为1.0。为了满足此要求，一开始的所有元素都会被初始化成1/m.
        D = mat(ones((5,1))/5)
        D1 = mat(ones((5,1)))
        print(D)
        print(D1)
        exit()
        bestStump, minError, bestClasEst = ada.buildStump(dataMat, classLabels, D)
        print('bestStump（字典）:', bestStump)
        print('minError（最小错误率）:', minError)
        print("bestClasEst（类别估计值）:", bestClasEst)
    elif ( type=="trainingAdaBoost" ):
        print("trainingAdaBoost")
        classifierArray = ada.adaBoostTrainDS(dataMat, classLabels, 9)

    elif ( type=='hardataset'):
        print('hardataset')
        datArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
        classifierArray  = ada.adaBoostTrainDS(datArr, labelArr, 10)

        testArr, testLabelArr = ada.loadDataSet('horseColicTest2.txt')
        prediction10 = ada.adaClassify(testArr, classifierArray)
        errArr = mat(ones((67,1)))
        errArrSum = errArr[prediction10 != mat(testLabelArr).T].sum()
        print("errArrSum:", errArrSum)
    elif ( type == 'plotroc' ):
        #ROC曲线的绘制及AUC计算函数
        datArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
        classifierArray, aggClassEst = ada.adaBoostTrainDS(datArr, labelArr)
        ada.plotROC(aggClassEst.T, labelArr)

    else:
        print("else")


