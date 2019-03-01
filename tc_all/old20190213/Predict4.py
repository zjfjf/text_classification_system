#!/usr/bin/python
# -*- coding: UTF-8 -*-
#import io
import os
#import pickle
import cPickle as pickle
#import tfidf
import Tools
import time
from sklearn.datasets.base import Bunch
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def get_stopword_list():
	# 停用词表存储路径，每一行为一个词，按行读取进行加载
	# 进行编码转换确保匹配准确率
	stop_word_path = '/usr/jyf/chinanews/分类test1/stopword.txt'
	stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,'r').readlines()]
	#stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path).readlines()]
	return stopword_list
	
def metrics_result(actual, predict):
	print('精确率:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
	print('召回率:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
	print('F1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
	
	
if __name__ == "__main__":
	
	starttime = time.time()
	print("starttime:" + str(starttime))  #starttime
	stopword_list = get_stopword_list()
	'''
	stop_word_path = '/usr/jyf/chinanews/分类test1/stopword.txt'
	stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,'r').readlines()]
	'''
	
	# 导入训练集
	train_tfidfspace= "/usr/jyf/chinanews/BOW/train_tfdifspace.dat"
	train_set = Tools.readbunchobj(train_tfidfspace)
	
	# 导入测试集
	test_tfidfspace = "/usr/jyf/chinanews/BOW/test_tfdifspace.dat"
	test_set = Tools.readbunchobj(test_tfidfspace)
	
	# 训练NB分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
	clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.labels)
	'''
	#LR
	clf=LogisticRegression(solver='lbfgs', multi_class = 'multinomial', max_iter=50)
	clf.fit(train_set.tdm, train_set.labels)
	'''
	
	'''
	#SVM 多分类器
	#clf = SVC(kernel='linear')  
	#model = OneVsRestClassifier(SVC(kernel='rbf'))   # default with 'rbf'  
	model = OneVsRestClassifier(SVC(kernel='linear'))
	clf =model.fit(train_set.tdm, train_set.labels)
	'''
	
	# 预测分类结果
	predicted = clf.predict(test_set.tdm)
	# 计算分类精度
	#metrics_result(sg_test_set.label, predicted)
	#metrics_result(fe_test_set.label, predicted)
	
	y_true = test_set.labels
	y_pred = predicted
	target_names = ['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
	print classification_report(y_true, y_pred, target_names=target_names)
	print(confusion_matrix(y_true, y_pred, labels=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']))
	
	'''
	PredictResult= 'F:/chinanews/BOW/PredictNB.txt'
	#tid = test_set.id
	tlabel = test_set.labels
	tpredicted = predicted
	f = io.open(PredictResult, "ab")
	k = 0
	for item1,item2 in zip(tlabel,tpredicted):
			
			tp= []
			str1 =str(item1)
			str2 =str(item2)
			tp.append(str(tid[k])) #文本ID
			tp.append(str1)  #文本真实类别label
			tp.append(str2)#文本预测类别predicted
			k = k+1
			
			Tools.savefileappend(PredictResult,','.join(tp).encode('utf-8'))
	f.close()
	'''
	endtime = time.time()
	print("endtime:" + str(endtime))  #endtime
	print ( "Time usage:" + str(endtime - starttime))  #执行时间
	
	
