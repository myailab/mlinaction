#!/usr/bin/python
import chapter2_knn.kNN as knn

if __name__ == "__main__":
    print("aaa")

    #手写识别系统测试
    #type = 'handwritingTest'


    #测试约会网站配对效果
    type = 'datingTest'

    if ( type == 'handwritingTest' ):
        knn.handwritingClassTest()
    elif ( type == 'datingTest' ):
        filePath = "testdata\\datingTestSet2.txt"
        knn.datingClassTest(filePath)
    else :
        print ('no matching')
