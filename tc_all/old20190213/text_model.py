#!/usr/bin/python
#encoding:utf-8

import  tensorflow as tf
import CBAMtest

class TextCNNConfig():

	embedding_size=200     #dimension of word embedding
	vocab_size=77216        #number of vocabulary
	pre_trianing = None   #use vector_char trained by word2vec

	seq_length= 100         #max length of sentence
	num_classes=8          #number of labels

	num_filters=128        #number of convolution kernel
	
	filter_sizes = [2, 3, 4]  #size of convolution kernel

	hidden_dim=128         #number of fully_connected layer units

	keep_prob=0.5          #droppout
	lr= 1e-3               #learning rate
	lr_decay= 0.9          #learning rate decay
	clip= 5.0              #gradient clipping threshold

	num_epochs=10         #epochs
	batch_size= 1280         #batch_size
	print_per_batch =100   #print result
	save_per_batch = 10  # 每多少轮存入tensorboard
	
	train_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/train.txt'  #train data
	test_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/test.txt'    #test data
	val_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/val.txt'      #validation data
	vocab_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/vocab.txt'        #vocabulary
	vector_word_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/vector_word.txt'  #vector_word trained by word2vec
	vector_word_npz='/usr/jyf/text_cnn_a/data/total_seg_fe2/vector_word.npz'   # save vector_word to numpy file
	
	
class TextCNN(object):

	def __init__(self,config):

		self.config=config

		self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
		self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
		self.keep_prob=tf.placeholder(tf.float32,name='dropout')
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		
		self.cnn()
	
	def cnn(self):
		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
											initializer=tf.constant_initializer(self.config.pre_trianing))
			#inputs=tf.nn.embedding_lookup(self.embedding,self.input_x)
			embedding_inputs = tf.nn.embedding_lookup(self.embedding,self.input_x)
			self.embedding_expand = tf.expand_dims(embedding_inputs, -1)
			#print(self.embedding_expand.get_shape())     # tuple   (?, 100, 200, 1)
			print(self.embedding_expand.shape.as_list()) # list      [None, 100, 200, 1]
			
			#CBAM module1 ，对词向量inputs加权
			self.embedding_expand_CBAM = CBAMtest.convolutional_block_attention_module(self.embedding_expand, 1, 0.5)
			print(self.embedding_expand_CBAM.shape.as_list())
		#Conv and Max-pooling
		## Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(self.config.filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
				w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
				b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
				conv = tf.nn.conv2d(self.embedding_expand_CBAM, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
				#Apply nonlinearity  relu
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(h, ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
								strides=[1, 1, 1, 1], padding='VALID', name='pool')
				print(pooled.shape.as_list()) #[None, 1, 1, 128]
				pooled_outputs.append(pooled)
		print("pooled_outputs")
		print(len(pooled_outputs))   # 3*[None, 1, 1, 128]
		
		
		#Combine all the pooled features
		num_filter_total = self.config.num_filters * len(self.config.filter_sizes)  #128*3
		self.h_pool = tf.concat(pooled_outputs, 3)
		print(self.h_pool.shape.as_list())  #[None, 1, 1, 384]
		
		#CBAM module2
		self.h_pool_attention= CBAMtest.convolutional_block_attention_module(self.h_pool, 2, 0.5)
		print(self.h_pool_attention.shape.as_list()) 
		#flat
		self.h_pool_flat = tf.reshape(self.h_pool_attention, [-1, num_filter_total])
		print(self.h_pool_flat.shape.as_list())  #[None, 384]
		
		
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
		
		#Final (unnormalized) scores and predictions
		with tf.name_scope('output'):
			w = tf.get_variable("w", shape=[num_filter_total, self.config.num_classes],
								initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
			#L2正则
			#l2_loss += tf.nn.l2_loss(W)
			#l2_loss += tf.nn.l2_loss(b)
			#wx+b
			self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")   
			#sofrmax(wx+b)
			self.pro = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.pro, 1, name="predictions")
				
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			#losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses)      #l2 reg
			#self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
			
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')
			
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.config.lr)
			gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
			gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
			#对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
			self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
			#global_step 自动+1
			
