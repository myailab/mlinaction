# @Time    : 2017/9/28 23:07
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm

import chapter7_adaboost.adaboost as ada
from numpy import *

if __name__ == '__main__':
    #type = "buildstump"
    #type = "trainingAdaBoost"
    type = "hardataset"

    dataMat, classLabels = ada.loadSimpData()

    if (type == "buildstump") :
        print("单层决策树生成函数：")
        D = mat(ones((5,1))/5)
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
    else:
        print("else")


