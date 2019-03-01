#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
#import jieba
#import jieba.posseg as psg
#from gensim import corpora, models
#from jieba import analyse
import functools
import io
import Tools
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 停用词表加载方法
def get_stopword_list():
	# 停用词表存储路径，每一行为一个词，按行读取进行加载
	# 进行编码转换确保匹配准确率
	stop_word_path = '/usr/jyf/chinanews/分类test1/stopword.txt'
	stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,'r').readlines()]
	#stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path).readlines()]
	return stopword_list

# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
	if not pos:
		# 不进行词性标注的分词方法
		seg_list = jieba.cut(sentence)
	else:
		# 进行词性标注的分词方法
		seg_list = psg.cut(sentence)
	return seg_list

# 去除干扰词
def word_filter(seg_list, pos=False):
	stopword_list = get_stopword_list()
	filter_list = []
	# 根据POS参数选择是否词性过滤
	## 不进行词性过滤，则将词性都标记为n，表示全部保留
	for seg in seg_list:
		if not pos:
			word = seg
			flag = 'n'
		else:
			word = seg.word
			flag = seg.flag
		if not flag.startswith('n'):
			continue
        # 过滤停用词表中的词，以及长度为<2的词
		if not word in stopword_list and len(word) > 1:
			filter_list.append(word)

	return filter_list

# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos, corpus_path='F:/SVM/chinese_text_classification/ST_train/C3-Art-train.txt'):
	# 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
	doc_list = []
	for line in open(corpus_path, 'r+'):
		content = line.strip()
		seg_list = seg_to_list(content, pos)
		filter_list = word_filter(seg_list, pos)
		doc_list.append(filter_list)

	return doc_list

# idf值统计方法
def train_idf(doc_list):
	idf_dic = {}
	# 总文档数
	tt_count = len(doc_list)

	# 每个词出现的文档数
	for doc in doc_list:
		for word in set(doc):
			idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

	# 按公式转换为idf值，分母加1进行平滑处理
	for k, v in idf_dic.items():
		idf_dic[k] = math.log(tt_count / (1.0 + v))

	# 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
	default_idf = math.log(tt_count / (1.0))
	return idf_dic, default_idf
	
#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
	import numpy as np
	res = np.sign(e1[1] - e2[1])
	if res != 0:
		return res
	else:
		a = e1[0] + e2[0]
		b = e2[0] + e1[0]
		if a > b:
			return 1
		elif a == b:
			return 0
		else:
			return -1
			
'''
# TF-IDF类
class TfIdf(object):
	# 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
	def __init__(self, idf_dic, default_idf, word_list, keyword_num):
		self.word_list = word_list
		self.idf_dic, self.default_idf = idf_dic, default_idf
		self.tf_dic = self.get_tf_dic()
		self.keyword_num = keyword_num

	# 统计tf值
	def get_tf_dic(self):
		tf_dic = {}
		for word in self.word_list:
			tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
			
		tt_count = len(self.word_list)
		for k, v in tf_dic.items():
			tf_dic[k] = float(v) / tt_count

		return tf_dic

	#按公式计算tf-idf
	#Returns:dict_feature_select:特征选择词字典
	def get_tfidf(self):
		tfidf_dic = {}
		for word in self.word_list:
			idf = self.idf_dic.get(word, self.default_idf)
			tf = self.tf_dic.get(word, 0)

			tfidf = tf * idf
			tfidf_dic[word] = tfidf
		#print(tfidf_dic)
		#tfidf_dic.items()
		# 根据tf-idf排序，去排名前keyword_num的词作为关键词
		for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
			print k + "/ ", end=''
		#dict_feature_select= sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]
		print "\n"
		#print( tfidf_dic.items() )
		print( tfidf_dic )
		return tfidf_dic 
		#return dict_feature_select
'''

def tfidf_extract(word_list, pos, keyword_num):
	#doc_list = load_data(pos)
	#idf_dic, default_idf = train_idf(doc_list)
	idf_dic, default_idf = train_idf(word_list)
	tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
	tfidf_model.get_tfidf()

def  corpus2file(path1,path2):
		#将短文本训练集分词、去停用词，写到一个file
		#return 列表 各文本预处理后的结果
		#path1 = "F:/SVM/chinese_text_classification/ST_train/C3-Art-train.txt"
		#path2 = "F:/SVM/chinese_text_classification/ST_train/C3-Art-train-seg.txt"
		list = []
		f1 = io.open(path1,mode = 'r+',encoding='utf-8')
		lines = f1.readlines()
		
		if lines :
			print "OK!"
		else:
			print "Error!"
		
		f2 = io.open(path2, mode = 'w+',encoding='utf-8')
		
		for line in lines:
			print "read success!"
			pos = True
			seg_list =  seg_to_list(line, pos)
			filter_list =  word_filter(seg_list, pos)
			content_seg = str(filter_list)
			list.append(filter_list)
			print(content_seg+"\n")
			f2.write(content_seg+"\n")
			print "write success!"
		f2.close()
		f1.close()
		return list
		
def  corpus2dir(path1,dir):
		#将短文本训练集分词、去停用词，写到一个dir
		#path1 = "F:/SVM/chinese_text_classification/ST_train/C3-Art-train.txt"
		#dir = "F:/SVM/chinese_text_classification/ST_train/C3-Art/"
		
		f1 = io.open(path1,mode = 'r+',encoding='utf-8')
		lines = f1.readlines()
		length = len(lines)
		
		if lines :
			print "OK!" 
		else:
			print "Error!" 
		
		i=0
		while(i <  length):
			for line in lines:
				print "read success!"
				#line = line.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()  # 删除换行
				#line = line.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()  # 删除空行、多余的空格
				#content_seg = jieba.cut(content)  # 为文件内容分词
				pos = True
				seg_list =  seg_to_list(line, pos)
				filter_list =  word_filter(seg_list, pos)
				content_seg = str(filter_list)
				
				path2 = dir + str(i+1)+".txt"
				f2 = io.open(path2, mode = 'w+',encoding='utf-8')
				f2.write(content_seg+"\n")
				f2.close()
				print "write success!"
				i = i + 1
			
		f1.close()
		
def  corpus2dir2(path1,dir):
		#将短文本训练集分词、去停用词，写到一个dir
		#path1 = "F:/SVM/chinese_text_classification/ST_train/C3-Art-train.txt"
		#dir = "F:/SVM/chinese_text_classification/ST_train/C3-Art/"
		
		f1 = open(path1,mode = 'r+',encoding='utf-8')
		lines = f1.readlines()
		length = len(lines)
		
		if lines :
			print "OK!"
		else:
			print "Error!"
		
		i=0
		while(i <  length):
			for line in lines:
				print "read success!"
				line = line.replace('\r\n', '').strip()  # 删除换行
				line = line.replace(' ', '').strip()  # 删除空行、多余的空格
				content_seg = jieba.cut(line)  # 为文件内容分词
				
				path2 = dir + str(i+1)+".txt"
				Tools.savefile(path2, ' '.join(content_seg))  # 将处理后的文件保存到分词后语料目录
				'''
				f2 = io.open(path2, mode = 'w+',encoding='utf-8')
				content_seg = str(content_seg)
				f2.write(content_seg)
				f2.close()
				'''
				print "write success!"
				i = i + 1
		f1.close()
		

			
		
		
		
		
		