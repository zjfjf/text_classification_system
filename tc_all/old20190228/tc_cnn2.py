#!/usr/bin/python
#encoding:utf-8

import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as kr
import tensorflow as tf
import numpy as np
import jieba
import time
import sys
import os
from collections import  Counter
from datetime import timedelta
from sklearn import metrics
from tc_datatype import *
from tc_data import *
from tc_tool import *

class cnn(object):
    
    def __init__(self,config):
        self.config=config

    def run(self,param):
        if param['data']['worktype']==worktype.train:
            return self.__do_trian(param)
        elif param['data']['worktype']==worktype.test:
            return self.__do_test(param)
        elif param['data']['worktype']==worktype.predict:
            return self.__do_pred(param)

###############################
    def __do_trian(self,param):
        dt=data(None).load(param['data'])
        self.config['vocab_size']=dt['wv_word_size']
        self.config['pre_trianing']=dt['wv_vector_table']
        x_train,y_train=dt['trainXYid_table']
        x_val,y_val=dt['valXYid_table']
        self.__init_param()        
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(param['cnn']['tensorboarddir'])
        session=tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        savepath=os.path.join(param['cnn']['savedir'], param['cnn']['savename'])
        if tool.has_file(param['cnn']['savedir'], param['cnn']['savename']):
            saver.restore(sess=session,save_path=savepath)        
        best_val_accuracy = 0
        last_improved = 0
        require_improvement = 200
        flag=False
        for epoch in range(self.config['num_epochs']):
            print('epoch={}'.format(epoch))
            batch_train = tool.batch_iter(x_train, y_train, self.config['batch_size'])
            start = tool.get_time()
            for x_batch, y_batch in batch_train:
                feed_dict = {self.input_x: x_batch,self.input_y: y_batch,self.keep_prob:self.config['keep_prob']}
                _, global_step, train_summaries, train_loss, train_accuracy = session.run([self.optimizer, self.global_step,merged_summary, self.loss,self.accuracy], feed_dict=feed_dict)
                if global_step % self.config['save_per_batch'] == 0:
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, global_step)
                if global_step % self.config['print_per_batch'] == 0:
                    end = tool.get_time()
                    val_loss, val_accuracy = self.__evaluate(session, x_val, y_val)
                    writer.add_summary(train_summaries, global_step)
                    if val_accuracy > best_val_accuracy:
                        saver.save(session, savepath)
                        best_val_accuracy = val_accuracy
                        last_improved=global_step
                        improved_str = '*'
                    else:
                        improved_str = ''
                    print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                                    global_step, train_loss, train_accuracy, val_loss, val_accuracy,(end - start) / self.config['print_per_batch'],improved_str))
                    start = tool.get_time()
                if global_step - last_improved > require_improvement:
                    print("No optimization over 200 steps, stop training")
                    flag = True
                    break
            if flag:
                    break
            self.config['lr'] *= self.config['lr_decay']
        return None

    def __do_test(self,param):
        dt=data(None).load(param['data'])        
        self.config['vocab_size']=dt['wv_word_size']
        self.config['pre_trianing']=dt['wv_vector_table']
        x_test,y_test=dt['testXYid_table']
        self.__init_param()
        session=tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        savepath=os.path.join(param['cnn']['savedir'], param['cnn']['savename'])
        saver.restore(sess=session,save_path=savepath)
        #test_loss, test_accuracy=self.__evaluate(session, x_test, y_test)        
        #print( 'Test Loss: {0}, Test Acc: {1}'.format(test_loss, test_accuracy))
        batch_size=self.config['batch_size']
        data_len=len(x_test)
        num_batch=int((data_len-1)/batch_size)+1
        y_test_cls=np.argmax(y_test,1)
        y_pred_cls=np.zeros(shape=len(x_test),dtype=np.int32)
            
        for i in range(num_batch):
            start_id=i*batch_size
            end_id=min((i+1)*batch_size,data_len)
            feed_dict={self.input_x:x_test[start_id:end_id],self.keep_prob:1.0}
            y_pred_cls[start_id:end_id]=session.run(self.predictions,feed_dict=feed_dict)
        report=metrics.classification_report(y_test_cls, y_pred_cls, target_names=param['data']['category'])
        mtx=metrics.confusion_matrix(y_test_cls, y_pred_cls)
        return report,mtx
            
    def __do_pred(self,param):
        dt=data(None).load(param['data'])        
        self.config['vocab_size']=dt['wv_word_size']
        self.config['pre_trianing']=dt['wv_vector_table']
        x_pred=dt['predXid_table'][0]
        self.__init_param()
        session=tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        savepath=os.path.join(param['cnn']['savedir'], param['cnn']['savename'])
        saver.restore(sess=session,save_path=savepath)
        feed_dict={self.input_x:x_pred,self.keep_prob:1.0}
        y_pred_cls=session.run(self.predictions,feed_dict=feed_dict)
        y_pred=[param['data']['category'][x] for x in y_pred_cls]
        return y_pred

###############################
    def __init_param(self):
        self.__set_input_x()
        self.__set_input_y()
        self.__set_keep_prob()
        self.__set_global_step()
        self.__set_l2loss()                             
        self.__set_embedding()
        self.__set_cov_max_pool()
        self.__set_dropout()
        self.__set_output()
        self.__set_loss()
        self.__set_accuracy()
        self.__set_optimizer()
        
    def __evaluate(self,sess, x_, y_):
        data_len = len(x_)
        batch_eval = tool.batch_iter(x_, y_, 128)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            feed_dict = {self.input_x:x_batch,self.input_y:y_batch,self.keep_prob:1.0}
            loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

###############################
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
            self.loss = tf.reduce_mean(losses)

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

def test_cnn_train():
    cnn_config,cnn_run_param={},{}
    for key,value in ex_cnnc_config.items():
        cnn_config[key]=value
    cnn_path={
                'trainfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\train.txt',
                'trainmidfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\trainmid.txt',
                'valfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\val.txt',
                'valmidfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\valmid.txt',
                'stopwordfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt',
                'wvfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt',
                'wv_wordfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt',
                'wv_vectorfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'}    
    cnn_data_param={
                'algorithmtype':algorithmtype.cnn,
                'worktype':worktype.train,
                'path':cnn_path,
                'pos':False,
                'category':['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融'],
                'max_length':100}
    cnn_={      'tensorboarddir':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\7\\',
                'savedir':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\7\\',
                'savename':'best_validation'}
    cnn_run_param['data']=cnn_data_param
    cnn_run_param['cnn']=cnn_
    _cnn=cnn(cnn_config)
    _cnn.run(cnn_run_param)
def test_cnn_test():
    cnn_config,cnn_run_param={},{}
    for key,value in ex_cnnc_config.items():
        cnn_config[key]=value
    cnn_path={
                'testfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\test_.txt',
                'testmidfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\testmid_.txt',
                'stopwordfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt',
                'wvfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt',
                'wv_wordfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt',
                'wv_vectorfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'}    
    cnn_data_param={
                'algorithmtype':algorithmtype.cnn,
                'worktype':worktype.test,
                'path':cnn_path,
                'pos':False,
                'category':['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融'],
                'max_length':100}
    cnn_={      'tensorboarddir':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\',
                'savedir':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\',
                'savename':'best_validation'}
    cnn_run_param['data']=cnn_data_param
    cnn_run_param['cnn']=cnn_
    _cnn=cnn(cnn_config)
    report,mtx=_cnn.run(cnn_run_param)
    print(report)
    print(mtx)
def test_cnn_pred():
    cnn_config,cnn_run_param={},{}
    for key,value in ex_cnnc_config.items():
        cnn_config[key]=value
    cnn_path={
                'predfile':['滴滴CEO程维跨年演讲：2016年将离出行梦想再近一步'],
                'predmidfile':'',
                #'predfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\pred.txt',
                #'predmidfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\predmid.txt',
                'stopwordfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt',
                'wvfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt',
                'wv_wordfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt',
                'wv_vectorfile':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'}    
    cnn_data_param={
                'algorithmtype':algorithmtype.cnn,
                'worktype':worktype.predict,
                'path':cnn_path,
                'pos':False,
                'category':['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融'],
                'max_length':100}
    cnn_={      'tensorboarddir':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\',
                'savedir':'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\',
                'savename':'best_validation'}
    cnn_run_param['data']=cnn_data_param
    cnn_run_param['cnn']=cnn_
    _cnn=cnn(cnn_config)
    pred=_cnn.run(cnn_run_param)
    print(pred)
if __name__ == "__main__":
    #test_cnn_train()
    #test_cnn_test()
    test_cnn_pred()
