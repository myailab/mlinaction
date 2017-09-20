'''
    第三章 决策树算法


'''

import chapter3_decisiontree.treePlotter as tp
import chapter3_decisiontree.trees as tr

if __name__ == "__main__":
    print("aaaa")

    #简单数据集绘制树形图
    type = "plot"

    #递规构建决策树
    #type = "trees"

    #计算香农熵
    #type = "shnnonEnt"

    #使用决策树预测隐形眼镜类型
    #type = "lenses"

    if ( type == "plot" ) :
        myTree = tp.retrieveTree (0)
        tp.createPlot(myTree)
        #print("plot")
    elif(type=="shnnonEnt"):
        #tr.calcShannonEnt()
        print("shnnonEnt")
    elif( type == "trees" ):
        myDat, labels = tr.createDataSet()
        myTree = tr.createTree(myDat,labels)
        print('myTree:', myTree)
    elif(type=="lenses"):
        fpath = "lenses.txt"
        fr = open(fpath, 'r')
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = tr.createTree(lenses, lensesLabels)
        tp.createPlot(lensesTree)

    else :
        print("bbbb")