#!/usr/bin/python
#encoding:utf-8

import  tensorflow as tf
import CBAMtest2
import CBAMtest


class TextCNNConfig():

	embedding_size=200     #dimension of word embedding
	vocab_size=77216        #number of vocabulary
	pre_trianing_c = None   #use vector_char trained by word2vec CBOW
	pre_trianing_s = None  #use vector_char trained by word2vec skipgram
	
	seq_length= 100         #max length of sentence
	num_classes=8          #number of labels

	num_filters=128        #number of convolution kernel
	
	filter_sizes = [2, 3]  #size of convolution kernel

	hidden_dim=128         #number of fully_connected layer units

	keep_prob=0.5          #droppout
	lr= 1e-3            #learning rate
	lr_decay= 0.9          #learning rate decay
	clip= 5.0              #gradient clipping threshold
	l2_reg_lambda = 0.001
	
	num_epochs=10         #epochs
	batch_size= 640         #batch_size
	print_per_batch =100   #print result
	save_per_batch = 10  # 每多少轮存入tensorboard
	
	
	train_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/train.txt'  #train data
	test_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/test.txt'    #test data
	val_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/val.txt'      #validation data
	vocab_filename='/usr/jyf/text_cnn_a/data/total_seg_fe2/vocab.txt'        #vocabulary
	
	vector_word_filename_c='/usr/jyf/text_cnn_a/data/total_seg_fe2/vector_word_c.txt'  #vector_word trained by word2vec
	vector_word_npz_c='/usr/jyf/text_cnn_a/data/total_seg_fe2/vector_word_c.npz'   # save vector_word to numpy file
	
	vector_word_filename_s ='/usr/jyf/text_cnn_a/data/total_seg_fe2/vector_word_s.txt'  #vector_word trained by word2vec
	vector_word_npz_s ='/usr/jyf/text_cnn_a/data/total_seg_fe2/vector_word_s.npz'   # save vector_word to numpy file
	

class TextCNN(object):
	def __init__(self,config):

		self.config=config

		self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
		self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
		self.keep_prob=tf.placeholder(tf.float32,name='dropout')
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		
		self.cnn()
	
	def cnn(self):
		# L2 loss
		self.l2_loss = tf.constant(0.0)
		
		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			
			self.embedding_r = tf.get_variable("embeddings_r", shape=[self.config.vocab_size, self.config.embedding_size])
			embedding_inputs_r = tf.nn.embedding_lookup(self.embedding_r, self.input_x)
			self.embedding_expand_r = tf.expand_dims(embedding_inputs_r, -1)
			print(self.embedding_expand_r.shape.as_list())# list      [None, 100, 200, 1]
			
			
			self.embedding_s = tf.get_variable("embeddings_s", shape=[self.config.vocab_size, self.config.embedding_size],
											initializer=tf.constant_initializer(self.config.pre_trianing_s))
			#inputs=tf.nn.embedding_lookup(self.embedding_s,self.input_x)
			embedding_inputs_s = tf.nn.embedding_lookup(self.embedding_s, self.input_x)
			self.embedding_expand_s = tf.expand_dims(embedding_inputs_s, -1)
			#print(self.embedding_expand_s.get_shape())     # tuple   (?, 100, 200, 1)
			print(self.embedding_expand_s.shape.as_list()) # list      [None, 100, 200, 1]
			
			self.embedding_c = tf.get_variable("embeddings_w", shape=[self.config.vocab_size, self.config.embedding_size],
											initializer=tf.constant_initializer(self.config.pre_trianing_c))
			#inputs=tf.nn.embedding_lookup(self.embedding_c,self.input_x)
			embedding_inputs_c = tf.nn.embedding_lookup(self.embedding_c, self.input_x)
			self.embedding_expand_c = tf.expand_dims(embedding_inputs_c, -1)
			#print(self.embedding_expand_c.get_shape())     # tuple   (?, 100, 200, 1)
			print(self.embedding_expand_c.shape.as_list()) # list      [None, 100, 200, 1]
			
			
			self.embedding_2 = tf.concat([self.embedding_expand_r, self.embedding_expand_s, self.embedding_expand_c], 3,name='embedding_2') #按axis=3拼接
			print(self.embedding_2.shape.as_list())# list      [None, 100, 200, 2]
			
			'''
			#CBAM module1 ，对词向量inputs加权
			self.embedding_2_CBAM = CBAMtest2.cbam_module(self.embedding_2, 1, 0.5)
			print(self.embedding_2_CBAM.shape.as_list())
			'''
			
			#CBAM module1 ，对词向量inputs加权
			self.embedding_2_CBAM = CBAMtest.convolutional_block_attention_module(self.embedding_2, 1, 0.5)
			print(self.embedding_2_CBAM.shape.as_list())
			#[<tf.Tensor 'embedding/cbam_1/strided_slice:0' shape=() dtype=int32>, 100, 200, 1]
			#[<tf.Tensor 'embedding/cbam_1/strided_slice:0' shape=() dtype=int32>, 100, 200, 2]
			#[<tf.Tensor 'embedding/cbam_1/strided_slice:0' shape=() dtype=int32>, 100, 200, 3]
		#Conv and Max-pooling
		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(self.config.filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, self.config.embedding_size, 3, self.config.num_filters]
				w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
				b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
				conv = tf.nn.conv2d(self.embedding_2_CBAM, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
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
		'''
		#CBAM module2
		self.h_pool_attention= CBAMtest2.cbam_module(self.h_pool, 2, 0.5)
		print(self.h_pool_attention.shape.as_list())
		'''
		
		'''
		#CBAM module2
		self.h_pool_attention= CBAMtest.convolutional_block_attention_module(self.h_pool, 2, 0.5)
		print(self.h_pool_attention.shape.as_list())  
		#[<tf.Tensor 'cbam_2/strided_slice:0' shape=() dtype=int32>, 1, 1, 384]
		'''
		#flat
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])
		print(self.h_pool_flat.shape.as_list())  #[None, 384]
		
		
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
		
		#Final (unnormalized) scores and predictions
		with tf.name_scope('output'):
			w = tf.get_variable("w", shape=[num_filter_total, self.config.num_classes],
								initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
			#L2正则
			self.l2_loss += tf.nn.l2_loss(w)
			self.l2_loss += tf.nn.l2_loss(b)
			
			#wx+b
			self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")   
			#sofrmax(wx+b)
			self.pro = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.pro, 1, name="predictions")
			
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			#losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
			#self.loss = tf.reduce_mean(losses)      #l2 reg
			self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * self.l2_loss
			
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')
			
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.config.lr)
			gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
			gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
			#对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g)得到新梯度
			self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
			#global_step 自动+1
			
