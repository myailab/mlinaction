import chapter4_bayes.bayes as bayes

if __name__ == "__main__":
    print("bayes")

    #使用朴素贝叶斯过滤垃圾邮件
    #type = "filterSpam"

    #帖子评论分类
    type = "postingClassify"


    if ( type == "filterSpam" ):
        bayes.spamTest()
    else:
        bayes.testingNB()