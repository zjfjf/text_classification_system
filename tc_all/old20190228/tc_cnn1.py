#!/usr/bin/python
#encoding:utf-8

import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as kr
import tensorflow as tf
import tkinter as tk
import numpy as np
import warnings
import codecs
import jieba
import time
import re
import sys
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from gensim.models import word2vec
from collections import  Counter
from datetime import timedelta
from sklearn import metrics
from tc_data import *

class tc_cnn(object):
    
    def __init__(self,tensorboard_dir,trained_modelfile,embedding_size=200,vocab_size=77216,pre_trianing=None,seq_length=100,num_classes=8,num_filters=128,
                 filter_sizes=[2,3,4],hidden_dim=128,keep_prob=0.5,lr=1e-3,lr_decay=0.9,clip=5.0,l2_reg_lambda=0.001,
                 num_epochs=10,batch_size=640,print_per_batch=100,save_per_batch=10):
        self.tensorboard_dir=tensorboard_dir
        self.trained_modelfile=trained_modelfile
        self.config={}
        self.config['embedding_size']=embedding_size##
        self.config['vocab_size']=vocab_size##
        self.config['pre_trianing']=pre_trianing##
        self.config['seq_length']=seq_length##
        self.config['num_classes']=num_classes
        self.config['num_filters']=num_filters
        self.config['filter_sizes']=filter_sizes
        self.config['hidden_dim']=hidden_dim
        self.config['keep_prob']=keep_prob
        self.config['lr']=lr
        self.config['lr_decay']=lr_decay
        self.config['clip']=clip
        self.config['l2_reg_lambda']=l2_reg_lambda
        self.config['num_epochs']=num_epochs
        self.config['batch_size']=batch_size##
        self.config['print_per_batch']=print_per_batch##
        self.config['save_per_batch']=print_per_batch##

    def init_param(self):
        input_x=self.__set_input_x()
        input_y=self.__set_input_y()
        keep_prob=self.__set_keep_prob()
        global_step = self.__set_global_step()
        l2_loss=self.__set_l2loss()                             
        self.__set_embedding()
        self.__set_cov_max_pool()
        self.__set_dropout()
        self.__set_output()
        self.__set_loss()
        self.__set_accuracy()
        self.__set_optimizer()

#
#instance_param
#input_x
#
    def __set_input_x(self):
        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config['seq_length']],name='input_x')
        return self.input_x

#
#instance_param
#input_y
#
    def __set_input_y(self):
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.config['num_classes']],name='input_y')
        return self.input_y

#
#instance_param
#keep_prob
# 
    def __set_keep_prob(self):
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')
        return self.keep_prob

#
#instance_param
#global_step
#
    def __set_global_step(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        return self.global_step

#
#instance_param
#l2_loss
#
    def __set_l2loss(self):
        self.l2_loss = tf.constant(0.0)
        return self.l2_loss
#
#instance_param
#embedding
#
    def __set_embedding_r(self):
        self.embedding_r = tf.get_variable("embeddings_r", shape=[self.config['vocab_size'], self.config['embedding_size']])
        return self.embedding_r
    
    def __set_embedding_expand_r(self,embedding_r):
        embedding_inputs_r = tf.nn.embedding_lookup(embedding_r, self.input_x)     
        self.embedding_expand_r = tf.expand_dims(embedding_inputs_r, -1)
        return self.embedding_expand_r
    
    def __set_embedding_s(self):
        self.embedding_s = tf.get_variable("embeddings_s", shape=[self.config['vocab_size'], self.config['embedding_size']],initializer=tf.constant_initializer(self.config['pre_trianing']))
        return self.embedding_s
    
    def __set_embedding_expand_s(self,embedding_s):
        embedding_inputs_s = tf.nn.embedding_lookup(embedding_s, self.input_x)
        self.embedding_expand_s = tf.expand_dims(embedding_inputs_s, -1)
        return self.embedding_expand_s

    def __set_embedding_2(self):
        self.embedding_2 = tf.concat([self.embedding_expand_r, self.embedding_expand_s], 3,name='embedding_2')
        return self.embedding_2
    
    def __set_embedding(self):        
        with tf.device('/cpu:0'), tf.name_scope('embedding'):     
            embedding_r = self.__set_embedding_r() 
            embedding_expand_r=self.__set_embedding_expand_r(embedding_r)	           
            embedding_s = self.__set_embedding_s()
            embedding_expand_s=self.__set_embedding_expand_s(embedding_s)
            embedding_2=self.__set_embedding_2()
            embedding_2_cbam = self.__set_cbam_()
#
#instance_param
#embedding
#cbam
#
    def __set_combined_static_and_dynamic_shape(self,tensor):
        static_tensor_shape = tensor.shape.as_list()
        dynamic_tensor_shape = tf.shape(tensor)
        combined_shape = []
        for index, dim in enumerate(static_tensor_shape):
            if dim is not None:
                combined_shape.append(dim)
            else:
                combined_shape.append(dynamic_tensor_shape[index])
        return combined_shape
    
    def __set_channel_weight_reshape(self,feature_map,feature_map_shape):
        channel_avg_weights = tf.nn.avg_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_max_weights = tf.nn.max_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_avg_reshape = tf.reshape(channel_avg_weights,[feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,[feature_map_shape[0], 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        return channel_w_reshape

    def __set_channel_attention(self,feature_map,feature_map_shape,channel_w_reshape,inner_units_ratio):
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=tf.nn.sigmoid
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        return feature_map_with_channel_attention
    
    def __set_channel_wise_reshape(self,feature_map_shape,feature_map_with_channel_attention):
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        return channel_wise_pooling

    def __set_convolution(self,channel_wise_pooling,feature_map_with_channel_attention):
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [3, 3],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention
    
    def __set_cbam_(self,index=1,inner_units_ratio=0.5):
        feature_map=self.embedding_2
        with tf.variable_scope("cbam_%s" % (index)):
            feature_map_shape = self.__set_combined_static_and_dynamic_shape(feature_map)
            channel_weight_reshape=self.__set_channel_weight_reshape(feature_map,feature_map_shape)
            feature_map_with_channel_attention=self.__set_channel_attention(feature_map,feature_map_shape,channel_weight_reshape,inner_units_ratio)
            channel_wise_reshape=self.__set_channel_wise_reshape(feature_map_shape,feature_map_with_channel_attention)
            feature_map_with_attention=self.__set_convolution(channel_wise_reshape,feature_map_with_channel_attention)
            self.embedding_2_cbam=feature_map_with_attention
            return self.embedding_2_cbam  
                
#
#instance_param
#cov_max_pool
#
    def __set_pool(self,filter_size):        
        filter_shape = [filter_size, self.config['embedding_size'], 2, self.config['num_filters']]
        w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[self.config['num_filters']]), name="b")
        conv = tf.nn.conv2d(self.embedding_2_cbam, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        pooled = tf.nn.max_pool(h, ksize=[1, self.config['seq_length'] - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name='pool')
        return pooled

    def __set_h_pool(self,pooled_outputs):
        self.h_pool = tf.concat(pooled_outputs, 3)
        return self.h_pool

    def __set_h_pool_flat(self):
        self.num_filter_total = self.config['num_filters'] * len(self.config['filter_sizes'])
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filter_total])
        return self.h_pool_flat
    
    def __set_cov_max_pool(self):  
        pooled_outputs = []
        for i, filter_size in enumerate(self.config['filter_sizes']):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                        pooled=self.__set_pool(filter_size)
                        pooled_outputs.append(pooled)
        h_pool = self.__set_h_pool(pooled_outputs)
        h_pool_flat=self.__set_h_pool_flat()

#
#instance_param
#dropout
#
    def __set_dropout(self):
        with tf.name_scope('dropout'):
            self.dropout = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
            
#
#instance_param
#output
#
    def __set_output(self):
        with tf.name_scope('output'):
            w = tf.get_variable("w", shape=[self.num_filter_total, self.config['num_classes']],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config['num_classes']]), name='b')
            self.l2_loss += tf.nn.l2_loss(w)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.dropout, w, b, name="scores")
            self.pro = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.pro, 1,name='predictions')

#
#instance_param
#loss
#
    def __set_loss(self):
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config['l2_reg_lambda'] * self.l2_loss

#
#instance_param
#accuracy
#
    def __set_accuracy(self):
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')

#
#instance_param
#optimizer
#
    def __set_optimizer(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config['lr'])
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config['clip'])
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

#
#train
#
    def __init_tf_sum(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.tensorboard_dir)  
                                       
    def __init_tf_sav(self,path=None):
        self.saver = tf.train.Saver()
        if path!=None:
            self.saver.restore(sess=self.session, save_path=path) 
                           
    def __init_tf_sess(self,train=True):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if train:
            self.writer.add_graph(self.session.graph)      

#
#train
#train_run
#
    def __init_mark(self):
        return 0,0,0,200,False

    def __evaluate(self,val_batch):
        batch_eval = val_batch
        total_loss = 0.0
        total_acc = 0.0
        total_len=0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            feed_dict ={self.input_x:x_batch,self.input_y:y_batch,self.keep_prob:1.0}
            loss, acc = self.session.run([self.loss, self.accuracy], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
            total_len += batch_len
        return total_loss / total_len, total_acc / total_len
    
    @staticmethod
    def get_time_dif(start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def __train_run(self,train_data,val_data):
        train_batch=train_data.batcher
        val_batch=val_data.batcher
        global_step,best_acc_val,last_improved,require_improvement,flag=self.__init_mark()
        start_time = time.time()                              
        for epoch in range(self.config['num_epochs']):
            for x_batch, y_batch in train_batch:                                       
                feed_dict = {self.input_x:x_batch,self.input_y:y_batch,self.keep_prob:self.config['keep_prob']}
                if global_step - last_improved > require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    flag = True  
                if global_step % self.config['save_per_batch'] == 0:
                    s = self.session.run(self.merged_summary, feed_dict=feed_dict)
                    self.writer.add_summary(s, global_step)                  
                if global_step % self.config["print_per_batch"] == 0:
                    feed_dict[self.keep_prob] = 1.0
                    loss_train, acc_train = self.session.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    loss_val, acc_val = self.__evaluate(val_batch)
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = global_step
                        self.saver.save(sess=self.session, save_path=self.trained_modelfile)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    time_dif = tc_cnn.get_time_dif(start_time)
                    print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},time:{},training speed: {:.3f}sec/batch {}\n".format(
						global_step, loss_train, acc_train, loss_val, acc_val,time_dif,time_dif.seconds / self.config['print_per_batch'],improved_str))
                self.session.run(self.optimizer, feed_dict=feed_dict)              
                global_step += 1
            if flag:
                break

#
#train
#test
#test_run
#
    def __test_run(self,test_data):        
        start_time = time.time()
        
        test_batch=test_data.load()
        loss_test, acc_test =self.__evaluate(test_batch)
        print('Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'.format(loss_test, acc_test))

        data_len = len(test_data.x_pad)
        num_batch = int((data_len - 1) / batch_size) + 1

        y_test_cls = np.argmax(test_data.y_pad, 1)
        y_pred_cls = np.zeros(shape=len(test_data.x_pad), dtype=np.int32)
        for i in range(num_batch):
            start_id = i * self.config['batch_size']
            end_id = min((i + 1) * self.config['batch_size'], data_len)
            feed_dict = {
                self.input_x: test_data.x_pad[start_id:end_id],
                self.keep_prob: 1.0
            }
            y_pred_cls[start_id:end_id] = self.session.run(self.predictions, feed_dict=feed_dict)
        
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=test_data.cat_iddic))
        print(metrics.confusion_matrix(y_test_cls, y_pred_cls))
        print("Time usage:", tc_cnn.get_time_dif(start_time))
#
#train
#test
#    
    def test(self,test_data):
        self.__init_tf_sess(train=False)
        self.__init_tf_sav(path=self.trained_modelfile)
        self.__test_run(test_data)
            
    def train(self,train_data,val_data,restore=False):                   
        self.__init_tf_sum()
        self.__init_tf_sess()
        self.__init_tf_sav(path=self.trained_modelfile if restore else None)
        self.__train_run(train_data,val_data)
#
#predict
#
    def predict(self,content):
        self.__init_tf_sess(train=False)
        self.__init_tf_sav(path=self.trained_modelfile)
        feed_dict = {
            self.input_x: content,
            self.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.predictions, feed_dict=feed_dict)
        return y_pred_cls[0]

if __name__ == "__main__":
    
    train_path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\train1\\train_jf.txt'
    #train_path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\3\\train.txt'
    var_path=train_path
    stop_path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\stopword.txt'
    train_data=tc_data(train_path,stop_path,batch_size=640,max_length=100,pos=False)
    train_batcher=train_data.load()

    tensorboard_dir='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\'
    trained_model_file=os.path.join(tensorboard_dir, 'best_validation')
    cnn=tc_cnn(tensorboard_dir,trained_model_file,vocab_size=len(train_data.model.wv.vectors),pre_trianing=train_data.model.wv.vectors,batch_size=640,print_per_batch=100,save_per_batch=10)
    print(cnn.config)    

    cnn.init_param()
    #ecnn.test(train_data)
    cnn.train(train_data,train_data,False)
    #data=train_data.content2id("泰达重启“魔鬼训练” 昆明气温回暖场地雪融")
    #print(train_data.categories[cnn.predict(data)])
