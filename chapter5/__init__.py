'''
    第五章 Logistic回归


'''


import chapter5.logRegress as lr

dataArr, labelMat = lr.loadDataSet()
weights = lr.gradAscent(dataArr, labelMat)

lr.plotBestFit(weights.getA())