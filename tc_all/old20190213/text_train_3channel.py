#!/usr/bin/python
#encoding:utf-8
from __future__ import print_function
from text_model import *
from loader import *
from sklearn import metrics
import sys
import os
import time
from datetime import timedelta
#from pyspark import  SparkContext, SparkConf

#APP_NAME = "pyspark word2vec cnn app"

def get_time_dif(start_time):
	"""获取已使用时间"""
	end_time = time.time()
	time_dif = end_time - start_time
	return timedelta(seconds=int(round(time_dif)))

#评估在某一数据上的准确率和损失
def evaluate(sess, x_, y_):
	data_len = len(x_)
	batch_eval = batch_iter(x_, y_, 128)
	total_loss = 0.0
	total_acc = 0.0
	for x_batch, y_batch in batch_eval:
		batch_len = len(x_batch)
		feed_dict = feed_data(x_batch, y_batch, 1.0)
		loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
		total_loss += loss * batch_len
		total_acc += acc * batch_len

	return total_loss / data_len, total_acc / data_len


def feed_data(x_batch, y_batch, keep_prob):
	feed_dict = {
		model.input_x: x_batch,
		model.input_y: y_batch,
		model.keep_prob:keep_prob
	}
	return feed_dict

def train():
	print("Configuring TensorBoard and Saver...")
	tensorboard_dir = '/usr/jyf/text_cnn_a/tensorboard/textcnn'
	if not os.path.exists(tensorboard_dir):
		os.makedirs(tensorboard_dir)
	
	save_dir = '/usr/jyf/text_cnn_a/checkpoints/textcnn'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, 'best_validation')
	
	print("Loading training and validation data...")
	start_time = time.time()
	
	x_train, y_train = process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
	x_val, y_val = process_file(config.val_filename, word_to_id, cat_to_id, config.seq_length)
	
	print(len(x_train))
	print(len(y_train))
	print(len(x_val))
	print(len(y_val))
	
	print("Time cost: %.3f seconds...\n" % (time.time() - start_time))
	
	tf.summary.scalar("loss", model.loss)
	tf.summary.scalar("accuracy", model.acc)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(tensorboard_dir)
	# 配置 Saver
	saver = tf.train.Saver()
	
	session = tf.Session()
	session.run(tf.global_variables_initializer())
	writer.add_graph(session.graph)
	
	print('Training and evaluating...')
	best_val_accuracy = 0
	last_improved = 0  # record global_step at best_val_accuracy
	require_improvement = 200  # break training if not having improvement over 1000 iter
	flag=False
	
	for epoch in range(config.num_epochs):
		batch_train = batch_iter(x_train, y_train, config.batch_size)
		#print(len(batch_train))
		'''
		n = 0
		for one in batch_train:
			 n = n +1
		print(n)
		'''
		start = time.time()
		print('Epoch:', epoch + 1)
		for x_batch, y_batch in batch_train:
			feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
			
			_, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
																				merged_summary, model.loss,
																				model.acc], feed_dict=feed_dict)
			print(global_step)
			print(train_loss)
			print(train_accuracy)
			
			if global_step % config.print_per_batch == 0:
				end = time.time()
				print("global_step % config.print_per_batch == 0")
				val_loss, val_accuracy = evaluate(session, x_val, y_val)
				writer.add_summary(train_summaries, global_step)
				
				# If improved, save the model
				if val_accuracy > best_val_accuracy:
					print("If improved, save the model")
					saver.save(session, save_path)
					best_val_accuracy = val_accuracy
					last_improved=global_step
					improved_str = '*'
				else:
					improved_str = ''
				print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
						global_step, train_loss, train_accuracy, val_loss, val_accuracy,
						(end - start) / config.print_per_batch,improved_str))
				start = time.time()

			if global_step - last_improved > require_improvement:
				print("No optimization over 200 steps, stop training")
				flag = True
				break
		#print(flag)
		if flag:
			break
		config.lr *= config.lr_decay

def mytrain():
	print("Configuring TensorBoard and Saver...")
	tensorboard_dir = '/usr/jyf/text_cnn_a/tensorboard/textcnn'
	if not os.path.exists(tensorboard_dir):
		os.makedirs(tensorboard_dir)
	
	save_dir = '/usr/jyf/text_cnn_a/checkpoints/textcnn'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, 'best_validation')
	
	print("Loading training and validation data...")
	start_time = time.time()
	
	x_train, y_train = myprocess_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
	x_val, y_val = myprocess_file(config.val_filename, word_to_id, cat_to_id, config.seq_length)
	'''
	print(len(x_train))
	print(len(y_train))
	print(len(x_val))
	print(len(y_val))
	'''
	
	print("Time cost: %.3f seconds...\n" % (time.time() - start_time))
	
	tf.summary.scalar("loss", model.loss)
	tf.summary.scalar("accuracy", model.accuracy)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(tensorboard_dir)
	# 配置 Saver
	saver = tf.train.Saver()
	
	session = tf.Session()
	session.run(tf.global_variables_initializer())
	writer.add_graph(session.graph)
	
	print('Training and evaluating...')
	best_val_accuracy = 0
	last_improved = 0  # record global_step at best_val_accuracy
	require_improvement = 100  # break training if not having improvement over 1000 iter
	
	flag=False
	for epoch in range(config.num_epochs):
		#batch_train = batch_iter(x_train, y_train, config.batch_size)
		batch_train = mybatch_iter(x_train, y_train, config.batch_size)
		start = time.time()
		print('Epoch:', epoch + 1)
		for x_batch, y_batch in batch_train:
			feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
			
			_, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optimizer, model.global_step,
																				merged_summary, model.loss,
																				model.accuracy], feed_dict=feed_dict)
			
			
			if global_step % config.save_per_batch == 0:
				# 每多少轮次将训练结果写入tensorboard scalar
				s = session.run(merged_summary, feed_dict=feed_dict)
				writer.add_summary(s, global_step)
			
			if global_step % config.print_per_batch == 0:
				end = time.time()
				
				# 每多少轮次输出在训练集和验证集上的性能
				#feed_dict[model.keep_prob] = 1.0
				#train_loss, train_accracy = session.run([model.loss, model.acc], feed_dict=feed_dict)
				val_loss, val_accuracy = evaluate(session, x_val, y_val)
				writer.add_summary(train_summaries, global_step)
				
				# If improved, save the model
				if val_accuracy > best_val_accuracy:
					saver.save(session, save_path)
					best_val_accuracy = val_accuracy
					last_improved=global_step
					improved_str = '*'
				else:
					improved_str = ''
				print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
						global_step, train_loss, train_accuracy, val_loss, val_accuracy,
						(end - start) / config.print_per_batch,improved_str))
				start = time.time()
			
			if global_step - last_improved > require_improvement:
				print("No optimization over 100 steps, stop training")
				flag = True
				break
		if flag:
			break
		config.lr *= config.lr_decay
		print("ok!")
	
if __name__ == '__main__':
	print('Configuring CNN model...')
	config = TextCNNConfig()
	
	#如果不存在词汇表，则构建词汇表
	if not os.path.exists(config.vocab_filename):
		#build_vocab(filenames, config.vocab_filename, config.vocab_size)
		#build_myvocab(config.train_filename, config.vocab_filename, config.vocab_size)
		#dictscores = build_myvocab1(config.train_filename, config.vocab_filename, config.vocab_size)
		#build_myvocab_w(config.train_filename, config.vocab_filename, config.vocab_size, train_tfidf_path)
		build_myvocab_all(config.train_filename, config.vocab_filename, config.vocab_size, config.vector_word_filename_s)
	
	#read vocab and categories
	categories,cat_to_id = read_mycategory()
	words,word_to_id = read_myvocab(config.vocab_filename)
	
	#词汇表长度
	config.vocab_size = len(words)
	print("vocab_size:"+ str(len(words)))
	
	#如果不存在embedding矩阵，则由词向量矩阵（txt）转化embedding矩阵（numpy数组）
	# trans vector file to numpy file
	if not os.path.exists(config.vector_word_npz_c):
		#export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
		export_word2vec_vectors_w(word_to_id, config.vector_word_filename_c, config.vector_word_npz_c)
	config.pre_trianing_c = get_training_word2vec_vectors(config.vector_word_npz_c)
	
	if not os.path.exists(config.vector_word_npz_s):
		#export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
		export_word2vec_vectors_w(word_to_id, config.vector_word_filename_s, config.vector_word_npz_s)
	config.pre_trianing_s = get_training_word2vec_vectors(config.vector_word_npz_s)
	
	model = TextCNN(config)
	mytrain()
	
	