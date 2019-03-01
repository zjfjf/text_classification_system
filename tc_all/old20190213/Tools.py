#!usr/bin/python  
# -*- coding:utf-8 _*-  

import pickle
#import cPickle as pickle
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)
# 追加保存至文件
def savefileappend(savepath, content):
    with open(savepath, "ab") as fp:
        fp.write(content)
        fp.write('\n'.encode('utf-8'))

# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

#写入bunch对象
def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch
