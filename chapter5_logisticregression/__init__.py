'''
    第五章 Logistic回归


'''


import chapter5_logisticregression.logRegress as lr

dataArr, labelMat = lr.loadDataSet()
weights = lr.gradAscent(dataArr, labelMat)

lr.plotBestFit(weights.getA())