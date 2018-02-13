# @Time    : 2018/2/10 13:19
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm

from numpy import *
import chapter10_kmeans.kMeans as kms


if __name__ == '__main__':
    # method = "test_methods"
    # method = "kmeans_test"
    method = "bisecting_kmeans"  # 二分K-均值算法
    # method = "condition_fpg"  # 创建条件FP树
    # method = "stagewise"

    # data_matrix, class_labels = ada.load_simple_data()

    if method == "test_methods":
        data_mat = mat(kms.load_dataset('testSet.txt'))
        min_0 = min(data_mat[:, 0])
        min_1 = min(data_mat[:, 1])
        max_0 = max(data_mat[:, 0])
        max_1 = max(data_mat[:, 1])
        print("第1列最小值：")
        print(min_0)
        print("第2列最小值：")
        print(min_1)
        print("第1列最大值：")
        print(max_0)
        print("第2列最大值：")
        print(max_1)
        print("生成min到max之间的值:")
        generate_val = kms.randCent(data_mat, 2)
        print(generate_val)
        print("测试距离计算的方法:")
        distance_val = kms.distEclud(data_mat[0], data_mat[1])
        print(distance_val)
    elif method == 'kmeans_test':
        data_mat = mat(kms.load_dataset('testSet.txt'))
        my_centroids, clust_assing = kms.kMeans(data_mat, 4)
        print(my_centroids)
        # print(clust_assing)
    elif method == 'bisecting_kmeans':
        data_mat = mat(kms.load_dataset('testSet2.txt'))
        centroid_list, my_new_assments = kms.biKmeans(data_mat, 3)
        print(centroid_list)

