import chapter4.bayes as bayes

if __name__ == "__main__":
    print("bayes")

    #使用朴素贝叶斯过滤垃圾邮件
    type = "filterSpam"


    if ( type == "filterSpam" ):
        bayes.spamTest()