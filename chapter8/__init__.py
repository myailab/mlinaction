# @Time    : 2018/1/10 18:10
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm

import chapter8.regression as reg
import matplotlib.pyplot as plt
from numpy import *

if __name__ == '__main__':
    # method = "stand_regression"
    # method = "lwlr"
    # method = "predict_age_of_abalone"
    method = "ridge_regression"

    # data_matrix, class_labels = ada.load_simple_data()

    if method == "stand_regression":
        print("执行标准的回归：")
        x_arr, y_arr = reg.loadDataSet('ex0.txt')
        ws = reg.standRegres(x_arr, y_arr)
        print(ws)
        # 绘制出数据集散点图和最佳拟合直线图：
        x_matrix = mat(x_arr)
        # print(x_matrix[1, :])
        # print(x_matrix[0:3, 1])  # [[ 0.067732][ 0.42781 ] [ 0.995731]]
        # print(x_matrix[0:3, 1].flatten())  # [[ 0.067732  0.42781   0.995731]]
        # print(x_matrix[0:3, 1].flatten().A[0])  # [ 0.067732  0.42781   0.995731]
        # exit()
        y_matrix = mat(y_arr)
        y_hat = x_matrix * ws
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_matrix[:, 1].flatten().A[0], y_matrix.T[:, 0].flatten().A[0])
        x_copy = x_matrix.copy()
        x_copy.sort(0)
        y_hat = x_copy * ws
        ax.plot(x_copy[:, 1], y_hat)
        plt.show()
    elif method == "lwlr":
        print("执行局部加权线性回归：")
        x_arr, y_arr = reg.loadDataSet('ex0.txt')
        ws = reg.lwlr(x_arr[0], x_arr, y_arr, 1.0)
        ws = reg.lwlr(x_arr[0], x_arr, y_arr, 0.001)
        y_hat = reg.lwlrTest(x_arr, x_arr, y_arr, 0.003)
        x_matrix = mat(x_arr)
        srtInd = x_matrix[:, 1].argsort(0)
        x_sort = x_matrix[srtInd][:, 0, :]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_sort[:, 1], y_hat[srtInd])
        ax.scatter(x_matrix[:, 1].flatten().A[0], mat(y_arr).T.flatten().A[0], s=2, c='red')
        plt.show()
    elif method == 'predict_age_of_abalone':
        print('预测鲍鱼的年龄：')
        abalone_x, abalone_y = reg.loadDataSet('abalone.txt')
        y_hat_01 = reg.lwlrTest(abalone_x[0:99], abalone_x[0:99], abalone_y[0:99], 0.1)
        y_hat_1 = reg.lwlrTest(abalone_x[0:99], abalone_x[0:99], abalone_y[0:99], 1)
        y_hat_10 = reg.lwlrTest(abalone_x[0:99], abalone_x[0:99], abalone_y[0:99], 10)
        y_hat_01_error = reg.rssError(abalone_y[0:99], y_hat_01)
        y_hat_1_error = reg.rssError(abalone_y[0:99], y_hat_1)
        y_hat_10_error = reg.rssError(abalone_y[0:99], y_hat_10)
        print("y_hat_01的误差为：%f" % y_hat_01_error)
        print("y_hat_1的误差为：%f" % y_hat_1_error)
        print("y_hat_10的误差为：%f" % y_hat_10_error)
    elif method == 'ridge_regression':
        # ROC曲线的绘制及AUC计算函数
        print("岭回归：")
        abalone_x, abalone_y = reg.loadDataSet('abalone.txt')
        ridge_weights = reg.ridgeTest(abalone_x, abalone_y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ridge_weights)
        plt.show()
    else:
        print("没有需要执行的程序")


