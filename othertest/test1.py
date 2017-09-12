from numpy import *

#创建一个五行三列的矩阵
returnMat = zeros((5,3))


#读取矩阵的行数
returnMat_shape0 = returnMat.shape[0]

#读取矩阵的行数和列数
returnMat_shape1 = returnMat.shape
print(returnMat)
print(returnMat_shape0)
print(returnMat_shape1)


a = array([[1,2], [3,4]])
b = array([[5,6], [7,8]])
m_a = mat(a)
m_b = mat(b)

print(m_a)
print(m_b)
print(m_a*m_b)
print(a*b)