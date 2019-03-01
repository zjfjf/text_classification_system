#!/usr/bin/python
#-*-coding:utf-8-*-
#from __future__ import division
import os
#os.environ["PYSPARK_PYTHON"]="/usr/bin/python"
#import io
#import pickle
import cPickle as pickle
from sklearn.datasets.base import Bunch
#from Tools import readfile
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#对seg_path下的文本构建Bunch对象，写入wordbag_path
def corpus2Bunch(wordbag_path, seg_path):
	catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
	# 创建一个Bunch实例
	bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
	bunch.target_name.extend(catelist)
	
	# 获取每个目录下所有的文件
	for file in catelist:
		fullname = seg_path + file  # 拼出文件名全路径
		f = io.open(fullname,'r+',encoding='utf-8')
		lines = f.readlines()
		for line in lines:
			bunch.label.append(file)
			bunch.filenames.append(fullname)
			bunch.contents.append(line) 
		# 将bunch存储到wordbag_path路径中
		
	with open(wordbag_path, "wb") as file_obj:
		pickle.dump(bunch, file_obj)
	print "构建文本对象结束！！！"
	

def corpus2Bunch1(wordbag_path, seg_path):
	
	catelist= ['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
	bunch = Bunch(target_name=[], labels=[],  contents=[])
	bunch.target_name.extend(catelist)
	
	f = open(seg_path, 'r+')
	lines = f.readlines()
	for line in lines:
		flabel, fcontent = line.split('\t')
		bunch.labels.append(flabel)
		bunch.contents.append(fcontent) 
	
	with open(wordbag_path, "wb") as file_obj:
		pickle.dump(bunch, file_obj)
	print "构建文本对象结束！！！"
	
	
if __name__ == "__main__":
	
	# 对训练集进行Bunch化操作:
	train_wordbag_path = "/usr/jyf/chinanews/BOW_fe/train_fe.dat"  # Bunch存储路径
	train_seg_path = "/usr/jyf/chinanews/total_seg_fe2/train.txt"  # 分词后分类语料库路径
	corpus2Bunch1(train_wordbag_path, train_seg_path)
	
	
	# 对测试集进行Bunch化操作：
	test_wordbag_path = "/usr/jyf/chinanews/BOW_fe/test_fe.dat"  # Bunch存储路径
	test_seg_path = "/usr/jyf/chinanews/total_seg_fe2/test.txt"  # 分词后分类语料库路径
	corpus2Bunch1(test_wordbag_path, test_seg_path)
	
	
	train_wordbag_path = "/usr/jyf/chinanews/BOW/train.dat"  # Bunch存储路径
	train_seg_path = "/usr/jyf/chinanews/seg1/train.txt"  # 分词后分类语料库路径
	corpus2Bunch1(train_wordbag_path, train_seg_path)
	
	test_wordbag_path = "/usr/jyf/chinanews/BOW/test.dat"  # Bunch存储路径
	test_seg_path = "/usr/jyf/chinanews/seg1/test.txt"  # 分词后分类语料库路径
	corpus2Bunch1(test_wordbag_path, test_seg_path)
	
	