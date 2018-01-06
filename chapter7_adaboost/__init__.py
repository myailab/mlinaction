# @Time    : 2017/9/28 23:07
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm

import chapter7_adaboost.adaboost as ada
from numpy import *

if __name__ == '__main__':
    method = "build_stump"
    # method = "training_adaboost"
    # method = "horse_colic_predict"
    # method = "plot_roc"

    data_matrix, class_labels = ada.load_simple_data()

    if method == "build_stump":
        print("单层决策树生成函数：")
        # D是一个概率分布向量，因此其所有的元素之和为1.0。为了满足此要求，一开始的所有元素都会被初始化成1/m.
        d = mat(ones((5, 1)) / 5)
        best_stump, min_error, best_class_estimate = ada.buildStump(data_matrix, class_labels, d)
        print('best_stump（字典）:', best_stump)
        print('min_error（最小错误率）:', min_error)
        print("best_class_estimate（最佳类别估计值）:", best_class_estimate)
    elif method == "training_adaboost":
        print("训练adaboost算法：")
        classifierArray = ada.adaBoostTrainDS(data_matrix, class_labels, 9)
    elif method == 'horse_colic_predict':
        print('马疝气病预测：')
        datArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
        classifierArray = ada.adaBoostTrainDS(datArr, labelArr, 10)
        testArr, testLabelArr = ada.loadDataSet('horseColicTest2.txt')
        prediction10 = ada.adaClassify(testArr, classifierArray)
        errArr = mat(ones((67, 1)))
        errArrSum = errArr[prediction10 != mat(testLabelArr).T].sum()
        print("errArrSum:", errArrSum)
    elif method == 'plot_roc':
        # ROC曲线的绘制及AUC计算函数
        print("绘制ROC曲线：")
        datArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
        classifierArray, aggClassEst = ada.adaBoostTrainDS(datArr, labelArr)
        ada.plotROC(aggClassEst.T, labelArr)
    else:
        print("没有需要执行的程序")


