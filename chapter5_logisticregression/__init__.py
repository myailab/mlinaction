'''
    第五章 Logistic回归


'''


import chapter5_logisticregression.logRegress as lr

pross_type = 'horse'

if __name__ == '__main__':
    if ( pross_type == 'plotbestfit' ) :
        #画出数据集和Logistic回归最佳拟合直线的函数
        dataArr, labelMat = lr.loadDataSet()
        weights = lr.gradAscent(dataArr, labelMat)

        lr.plotBestFit(weights.getA())
    elif ( pross_type == 'horse' ) :
        '''
        从疝气病症预测马的死亡率
        '''
        print('horse')
        lr.multiTest()