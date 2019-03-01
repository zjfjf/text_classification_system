#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import warnings
import time
import codecs
import sys
import re
#import jieba
from gensim.models import word2vec
from text_model import TextConfig
import loader
import Tools

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
#将每个文本转成Sentence List列表
def CorpusToSentenceList(train_path):
	train, label = loader.read_myfile(train_path)
	sentencelist = []   #sentencelist, 每个元素是word list
	for line in train:
		#line = line.decode('utf-8')
		line = line.replace('\r\n', '').strip()
		line = line.split(',')
		sentencelist.append(line)
	return sentencelist

re_han= re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)") # the method of cutting text by punctuation

class Get_Sentences(object):
	'''
	Args:
		filenames: a list of train_filename,test_filename,val_filename
	Yield:
		word:a list of word cut by jieba
	'''

	def __init__(self,filenames):
		self.filenames= filenames

	def __iter__(self):
		for filename in self.filenames:
			with codecs.open(filename, 'r', encoding='utf-8') as f:
				for _,line in enumerate(f):
					try:
						line=line.strip()
						line=line.split('\t')
						assert len(line)==2
						blocks=re_han.split(line[1])
						word=[]
						for blk in blocks:
							if re_han.match(blk):
								word.extend(jieba.lcut(blk))
						yield word
					except:
						pass

def train_word2vec(filenames):
	'''
	use word2vec train word vector
	argv:
		filenames: a list of train_filename,test_filename,val_filename
	return: 
		save word vector to config.vector_word_filename
	'''
	t1 = time.time()
	sentences = Get_Sentences(filenames)
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=2, workers=4)
	model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
	print('-------------------------------------------')
	print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

def train_myword2vec(file_path):
	'''
	use word2vec train word vector
	argv:
		filenames: a list of train_filename,test_filename,val_filename
	return: 
		save word vector to config.vector_word_filename
	'''
	t1 = time.time()
	#sentences = Get_Sentences(filenames)
	sentences =  CorpusToSentenceList(file_path) #list
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = word2vec.Word2Vec(sentences, sg=1,hs=1,size=200, window=1, min_count=5, sample=0.001, negative=5, workers=4)
	#model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
	model.save_word2vec_format(config.vector_word_filename, binary=False)
	print('------------------------------------')
	print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

def train_model(filepath, modelpath):
	fpath = filepath
	mpath = modelpath
	sentences =  CorpusToWordList(fpath) #list
	model = word2vec.Word2Vec(sentences, sg=1,hs=1,min_count=5,window=3,size=200, negative=3, sample=0.001,workers=4)
	model.save_word2vec_format(mpath,binary=True)

def get_model(modelpath):
	mpath = modelpath
	model= word2vec.Word2Vec.load_word2vec_format(mpath, binary=True)
	return model
	
if __name__ == '__main__':
	 
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	config=TextConfig()
	'''
	filenames=[config.train_filename,config.test_filename,config.val_filename]
	train_word2vec(filenames)
	'''
	filepath1 = "/usr/jyf/chinanews/total_seg/total_seg.txt"
	
	#train and  save model
	train_myword2vec(filepath1)
	
	#load model
	#model = get_model(modelpath1)
	