# @Time    : 2017/9/21 23:47
# @Author  : myailab
# @Site    : http://www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm
#!/usr/bin/python
import chapter11_apriori.apriori as ap

if __name__ == "__main__":
    print("aaa")

    #手写识别系统测试
    #type = 'handwritingTest'


    #测试约会网站配对效果
    type = 'datingTest'

    if ( type == 'handwritingTest' ):
        print('apriori')
        # knn.handwritingClassTest()
    elif ( type == 'datingTest' ):
        print('apriori')
        # filePath = "testdata\\datingTestSet2.txt"
        # knn.datingClassTest(filePath)
    else :
        print ('no matching')