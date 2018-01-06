"""
    记录自定义日志的模块
    Data：2017-08-10
    @author:zhangyong
"""
import datetime
import io


class Syslog:
    def __init__(self):
        self.suffix = '.log'
        print('syslog')

    def saveLog(self, log_str_params):
        # 获取日期，并在目录中查找是否有这个日期的文件，如果没有则自动创建一个，然后将日志内容保存
        now = datetime.datetime.now()
        date = now.strftime('%Y-%m-%d')
        log_filename = date+self.suffix
        file_o = open(log_filename, 'a+')
        log_str = date+':'+str(log_str_params)+"\n"
        file_o.write(log_str)
        file_o.close()
        return True
