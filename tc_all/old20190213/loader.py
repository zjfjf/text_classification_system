#!/usr/bin/python
#encoding:utf-8

from collections import  Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import sys
#import jieba
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import Tools


if sys.version_info[0] > 2:
	is_py3 = True
else:
	reload(sys)
	sys.setdefaultencoding("utf-8")
	is_py3 = False

def native_word(word, encoding='utf-8'):
	"""如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
	if not is_py3:
		return word.encode(encoding)
	else:
		return word

def native_content(content):
	if not is_py3:
		return content.decode('utf-8')
	else:
		return content
	
def open_file(filename, mode='r'):
	"""
	常用文件操作，可在python2和python3间切换.
	mode: 'r' or 'w' for read or write
	"""
	if is_py3:
		return open(filename, mode, encoding='utf-8', errors='ignore')
	else:
		return open(filename, mode)
	
def read_file(filename):
	"""
	Args:
		filename:trian_filename,test_filename,val_filename 
	Returns:
		two list where the first is lables and the second is contents cut by jieba
	"""
	re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
	contents,labels=[],[]
	with codecs.open(filename,'r',encoding='utf-8') as f:
		for line in f:
			try:
				line=line.rstrip()
				assert len(line.split('\t'))==2
				label,content=line.split('\t')
				labels.append(label)
				blocks = re_han.split(content)
				word = []
				for blk in blocks:
					if re_han.match(blk):
						word.extend(jieba.lcut(blk))
				contents.append(word)
			except:
				pass
	return labels,contents

def read_myfile(filename):
	"""读取文件数据"""
	contents, labels = [], []
	with open_file(filename) as f:
		for line in f:
			try:
				label, content = line.strip().split('\t')
				if content:
					contents.append((native_content(content)))
					labels.append(native_content(label))
			except:
				pass
	return contents, labels
	
def build_vocab(filenames,vocab_dir,vocab_size=8000):
	"""
	Args:
		filename:trian_filename,test_filename,val_filename
		vocab_dir:path of vocab_filename
		vocab_size:number of vocabulary
	Returns:
		writting vocab to vocab_filename
	"""
	all_data = []
	for filename in filenames:
		_,data_train=read_file(filename)
		for content in data_train:
			all_data.extend(content)
	counter=Counter(all_data)
	words,_=list(zip(*count_pairs))
	words=['<PAD>']+list(words)

	with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
		f.write('\n'.join(words)+'\n')

def build_myvocab(train_dir, vocab_dir, vocab_size):
	"""根据训练集构建词汇表，存储"""
	data_train, _ = read_myfile(train_dir)
	all_data = []
	for line in data_train:
		#line = line.decode('utf-8')
		line = line.replace('\r\n', '').strip()    # 删除换行
		line = line.split(',')
		all_data.extend(line)
	
	counter = Counter(all_data)
	count_pairs = counter.most_common(vocab_size - 1)  #key:单词，value:出现次数
	words, _ = list(zip(*count_pairs))  #解压，取key
	# 添加一个 <PAD> 来将所有文本pad为同一长度
	words = ['<PAD>'] + list(words)
	open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
	
def build_myvocab_w(train_dir, vocab_dir, vocab_size, train_tfidf_path):
	"""根据训练集构建词汇表，存储"""
	
	if train_tfidf_path is not None:        
		trainbunch = Tools.readbunchobj(train_tfidf_path)
		words = trainbunch.vocabulary  #导入训练集的TF-IDF词向量空间
	
	'''
	#chi 选择特征
	train_np = np.array(trainbunch.tdm)
	label_np = np.array(trainbunch.labels)
	
	model1 = SelectKBest(chi2, k = vocab_size )  #选择k个最佳特征
	words = model1.fit_transform(train_np, label_np)#选择出k个特征 
	scores = model1.scores_  #得分
	'''
	
	# 添加一个 <PAD> 来将所有文本pad为同一长度
	words = ['<PAD>'] + list(words)
	open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
	
	'''
	normalization_values = values/max(values)
	count_dict = dict(zip(words, normalization_values))
	return count_dict
	'''
	
def build_myvocab_all(train_dir, vocab_dir, vocab_size, word2vec_path):
	"""根据训练集构建词汇表，存储"""
	words = []
	file_r = codecs.open(word2vec_path, 'r', encoding='utf-8')
	line = file_r.readline()
	voc_size, vec_dim = map(int, line.split(' ')) #word2vec的单词总数，词向量维度
	line = file_r.readline()
	while line:
		try:
			items = line.split(' ')
			word = items[0]  #单词
			words.append(word)
		except:
			pass
		line = file_r.readline()
	
	# 添加一个 <PAD> 来将所有文本pad为同一长度
	words = ['<PAD>'] + list(words)
	open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
	
	'''
	normalization_values = values/max(values)
	count_dict = dict(zip(words, normalization_values))
	return count_dict
	'''
	
def build_myvocab1(train_dir, vocab_dir, vocab_size):
	"""根据训练集构建词汇表，存储"""
	train, label = read_myfile(train_dir)
	all_data = []
	for line in train:
		#line = line.decode('utf-8')
		line = line.replace('\r\n', '').strip()    # 删除换行
		line = line.split(',')
		all_data.append(line)
	print("all_data")
	print( len(all_data) )
	
	train_np = np.array(all_data)
	label_np = np.array(label)
	
	print("train_np.shape")
	print(train_np.shape)
	print("label_np.shape")
	print(label_np.shape)
	
	model1 = SelectKBest(chi2, k = vocab_size )  #选择k个最佳特征
	words = model1.fit_transform(train_np, label_np)#该函数可以选择出k个特征 
	scores = model1.scores_  #得分
	
	words = words.tolist()
	scores = scores.tolist()
	
	print(len(words))
	print(len(scores))
	
	dictscores = dict(zip(words, scores))
	
	# 添加一个 <PAD> 来将所有文本pad为同一长度
	words = ['<PAD>'] + list(words)
	open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
	
	return dictscores
	
def chi(x_train, y_train, feature_size):
	train_np = np.array(x_train)
	label_np = np.array(y_train)
	
	print("train_np.shape")
	print(train_np.shape)
	print("label_np.shape")
	print(label_np.shape)
	
	model1 = SelectKBest(chi2, k = feature_size )  #选择k个最佳特征
	words = model1.fit_transform(train_np, label_np)#该函数可以选择出k个特征 
	scores = model1.scores_  #得分
	
	words = words.tolist()
	scores = scores.tolist()
	
	print(len(words))
	print(len(scores))
	
	dictscores = dict(zip(words, scores))
	
def read_vocab(vocab_dir):
	"""
	Args:
	filename:path of vocab_filename
	Returns:
	words: a list of vocab
	word_to_id: a dict of word to id
	"""
	words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
	word_to_id=dict(zip(words,range(len(words))))
	return words,word_to_id

def read_myvocab(vocab_dir):
	"""读取词汇表"""
	# words = open_file(vocab_dir).read().strip().split('\n')
	with open_file(vocab_dir) as fp:
	# 如果是py2 则每个值都转化为unicode
		words = [native_content(_.strip()) for _ in fp.readlines()]
		word_to_id = dict(zip(words, range(len(words))))
	return words, word_to_id
def read_category():
	"""
	Args:
		None
	Returns:
		categories: a list of label
		cat_to_id: a dict of label to id
	"""
	categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
	
	cat_to_id=dict(zip(categories,range(len(categories))))
	return categories,cat_to_id
	
def read_mycategory():
	"""读取分类目录，固定"""
	#categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
	categories = ['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
	categories = [native_content(x) for x in categories]
	cat_to_id = dict(zip(categories, range(len(categories))))
	return categories, cat_to_id
	
def process_file(filename,word_to_id,cat_to_id,max_length=600):
	"""
	Args:
		filename:train_filename or test_filename or val_filename
		word_to_id:get from def read_vocab()
		cat_to_id:get from def read_category()
		max_length:allow max length of sentence 
	Returns:
		x_pad: sequence data from  preprocessing sentence 
		y_pad: sequence data from preprocessing label
	"""
	labels,contents=read_file(filename)
	data_id,label_id=[],[]
	for i in range(len(contents)):
		data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
		label_id.append(cat_to_id[labels[i]])
	x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
	y_pad=kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
	return x_pad,y_pad

def myprocess_file(filename, word_to_id, cat_to_id, max_length):
	"""将文件转换为id表示"""
	contents, labels = read_myfile(filename)
	print(len(contents))
	print(len(labels))
	'''
	for i in cat_to_id:
		print(i+ '\n')
	'''
	data_id, label_id = [], []
	for i in range(len(contents)):
		data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
		label_id.append(cat_to_id[labels[i]])
	
	# 使用keras提供的pad_sequences来将文本pad为固定长度
	x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
	y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
	return x_pad, y_pad
	
def batch_iter(x, y, batch_size=64):
	"""
	Args:
		x: x_pad get from def process_file()
		y:y_pad get from def process_file()
	Yield:
		input_x,input_y by batch size
	"""
	data_len=len(x)
	num_batch=int((data_len-1)/batch_size)+1

	indices=np.random.permutation(np.arange(data_len))
	x_shuffle=x[indices]
	y_shuffle=y[indices]

	for i in range(num_batch):
		start_id=i*batch_size
		end_id=min((i+1)*batch_size,data_len)
		yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]

def mybatch_iter(x, y, batch_size=64):
	"""生成批次数据"""
	data_len = len(x)
	num_batch = int((data_len - 1) / batch_size) + 1

	indices = np.random.permutation(np.arange(data_len))
	x_shuffle = x[indices]
	y_shuffle = y[indices]

	for i in range(num_batch):
		start_id = i * batch_size
		end_id = min((i + 1) * batch_size, data_len)
		yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

#将词向量矩阵（txt）转化为numpy file
def export_word2vec_vectors(vocab, word2vec_path,trimmed_filename):
	"""
	Args:
		vocab: word_to_id 
		word2vec_path: file path of have trained word vector by word2vec
		trimmed_filename: file path of changing word_vector to numpy file
	Returns:
		save vocab_vector to numpy file
	"""
	file_r = codecs.open(word2vec_path, 'r', encoding='utf-8')
	line = file_r.readline()
	voc_size, vec_dim = map(int, line.split(' ')) #word2vec的单词总数，词向量维度
	embeddings = np.zeros([len(vocab), vec_dim])  #embedding矩阵初始化为0  len(vocab)*vec_dim
	line = file_r.readline()
	while line:
		try:
			items = line.split(' ')
			word = items[0]  #单词
			vec = np.asarray(items[1:], dtype='float32')  #词向量
			if word in vocab: #如果word在词汇表vocab中
				word_idx = vocab[word] #word对应的id
				embeddings[word_idx] = np.asarray(vec)  #将embeddings矩阵word id对应的一行由0改为词向量
		except:
			pass
		line = file_r.readline()
	np.savez_compressed(trimmed_filename, embeddings=embeddings)#将embeddings矩阵存储为numpy数组

#将词向量矩阵（txt）转化为numpy file
def export_word2vec_vectors_w(vocab,  word2vec_path, trimmed_filename):
	"""
	Args:
		vocab: word_to_id 
		word2vec_path: file path of have trained word vector by word2vec
		trimmed_filename: file path of changing word_vector to numpy file
	Returns:
		save vocab_vector to numpy file
	"""
	file_r = codecs.open(word2vec_path, 'r', encoding='utf-8')
	line = file_r.readline()
	voc_size, vec_dim = map(int, line.split(' ')) #word2vec的单词总数，词向量维度
	embeddings = np.zeros([len(vocab), vec_dim])  #embedding矩阵初始化为0  len(vocab)*vec_dim
	line = file_r.readline()
	while line:
		try:
			items = line.split(' ')
			word = items[0]  #单词
			vec = np.asarray(items[1:], dtype='float32')  #词向量
			if word in vocab: #如果word在词汇表vocab中
				word_idx = vocab[word] #word对应的id
				#score = dictscores[word] #word对应的chi score
				embeddings[word_idx] = np.asarray(vec)  #将embeddings矩阵word id对应的一行由0改为词向量
				
		except:
			pass
		line = file_r.readline()
	#考虑上下文位置，t-1,t,t+1 相加求平均
	np.savez_compressed(trimmed_filename, embeddings=embeddings)#将embeddings矩阵存储为numpy数组
	
def get_training_word2vec_vectors(filename):
	"""
	Args:
		filename:numpy file
	Returns:
		data["embeddings"]: a matrix of vocab vector
	"""
	with np.load(filename) as data:
		return data["embeddings"]
