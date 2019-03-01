#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as kr
import tensorflow as tf
import tkinter as tk
import numpy as np
import warnings
import logging
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
from enum import IntEnum

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

class datatype(IntEnum):
	file=0,
	string=1

class algorithmtype(IntEnum):
	cnn=0,
	nb=1,
	lr=2

class worktype(IntEnum):
	train=0,
	test=1,
	predict=2
	
class predtype(IntEnum):
	file=0,
	string=1
	
ex_cnn_train_data={
                'trainidXYid_table':None,
                'validXY_table':None,
                'wv_word_size':None,
                'wv_vector_table':None}

ex_cnn_test_data={
                'testidXY_table':None,
                'wv_word_size':None,
                'wv_vector_table':None}

ex_cnn_pred_data={
                'predidX_table':None,              
                'wv_word_size':None,
                'wv_vector_table':None}

ex_data_config={
                'path':None,
                'pos':None,
                'category':None}

ex_cnn_train_path={
                'trainfile':None,
                'trainmidfile':None,
                'valfile':None,
                'valmidfile':None,
                'stopwordfile':None,
                'wvfile':None,
                'wv_wordfile':None,
                'wv_vectorfile':None}

ex_cnn_test_path={
                'testfile':None,
                'testmidfile':None,
                'stopwordfile':None,
                'wvfile':None,
                'wv_wordfile':None,
                'wv_vectorfile':None}

ex_cnn_pred_path={
                'stopwordfile':None,
                'wvfile':None,
                'wv_wordfile':None,
                'wv_vectorfile':None}

ex_word2vec_param={}

ex_cnn_param={
                'worktype':None,
                'path':None,
                'pos':None,
                'category':None,
                'max_length':None,
                'word2vec':None,
                'pend1':None,
                'pend2':None}

	
class data(object):
    def __init__(self,config):
        self.config=config

    def __get_stopword(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            return [sw.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\n', '') for sw in f.readlines()]

    def __gen_sentence(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

    def __format_sentence(self,sentence):
        st=sentence.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\r','').replace('\n', '')
        return st if '\t' in st else st[:2]+'\t'+st[2:]

    def __cut_sentence(self,sentence,pos):
        return psg.cut(sentence) if pos else jieba.cut(sentence)
    
    def __filter_word(self,wordlist,stopwordlist,pos):
        filter_list=[]
        for seg in wordlist:
            if not pos:
                word = seg
                flag = 'n'
            else:
                word = seg.word
                flag = seg.flag
            if not flag.startswith('n'):
                continue
            if not word in stopwordlist:
                if word.isspace() and word == '\t':
                    filter_list.append(word)
                if not word.isspace():
                    filter_list.append(word)
        return filter_list

    def __format_wordsentence(self,wordlist):
        cl,ct=','.join(wordlist).split('\t,')
        return cl.replace(',','')+'\t'+ct

    def __file2midfile(self,path,midpath,stopwordpath,pos):
        wordsentencelist=[]
        sw=self.__get_stopword(stopwordpath)
        sentence=self.__gen_sentence(path)
        for s in sentence:
            fs=self.__format_sentence(s)
            cs=self.__cut_sentence(fs,pos)
            fw=self.__filter_word(cs,sw,pos)
            fws=self.__format_wordsentence(fw)
            wordsentencelist.append(fws)
        with open(midpath,'w',encoding='utf-8') as f:
            f.write('\n'.join(wordsentencelist)+'\n')
            
    def __word2vec(self,path,savepath,param=None):
        wordsentence=self.__gen_sentence(path)
        wordsentence_=[]
        wordlist=[]
        for ws in wordsentence:
            _, wsc = ws.strip().split('\t')
            wordsentence_.append(wsc)
        for ws_ in wordsentence_:
            wordlist.append(ws_.split(','))
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = word2vec.Word2Vec(wordlist, sg=1,hs=1,size=200, window=1, min_count=5, sample=0.001, negative=5, workers=4)
        model.wv.save_word2vec_format(savepath, binary=False)
    
    def __wv2word(self,wvpath,savepath):
        word=[]
        sentence=self.__gen_sentence(wvpath)
        next(sentence)
        for st in sentence:
            s=st.split(' ')
            word.append(s[0])
        w = ['<PAD>'] + list(word)
        with open(savepath,'w',encoding='utf-8') as f:
            f.write('\n'.join(w) + '\n')
        return w
    
    def __get_wv_word_id(self,wvpath,savepath):
        word=None
        if os.path.exists(savepath):
            sentence=self.__gen_sentence(savepath)
            word=[s.strip() for s in sentence]
        else:
            word=self.__wv2word(wvpath,savepath)
        return dict(zip(word, range(len(word))))

    def __get_category_id(self,category):
        return dict(zip(category, range(len(category))))
    
    def __wv2vector(self,wv_word_id,wvpath,savepath):
        sentence=self.__gen_sentence(wvpath)
        size,dim=map(int,next(sentence).split(' '))
        embedding = np.zeros([len(wv_word_id),dim])
        for st in sentence:
            s=st.split(' ')
            word = s[0].strip()
            vector = np.asarray(s[1:], dtype='float32')
            if word in wv_word_id:
                wid = wv_word_id[word]
                embedding[wid] = np.asarray(vector)
        np.savez_compressed(savepath, embeddings=embedding)
        return embedding
        
    def __get_wv_vector(self,wv_word_id,wvpath,savepath):
        vector=None
        if os.path.exists(savepath):
            with np.load(savepath) as data:
                vecotr=data["embeddings"]
        else:
            vecotr=self.__wv2vector(wv_word_id,wvpath,savepath)
        return vector

    def __get_XYidtable(self,path,wv_word_id,wv_category_id,max_length):
        data_id, label_id = [], []
        sentence=self.__gen_sentence(path)
        for st in sentence:
            label,content=st.strip().split('\t')
            data_id.append([wv_word_id[x] for x in content if x in wv_word_id])
            label_id.append(wv_category_id[label.encode('utf-8').decode('utf-8-sig')])
        x = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y = kr.utils.to_categorical(label_id, num_classes=len(wv_category_id))
        return x,y
        
    def __cnn_train(self,param):     
        self.__file2midfile(param['path']['trainfile'],param['path']['trainmidfile'],param['path']['stopwordfile'],False)
        self.__file2midfile(param['path']['valfile'],param['path']['valmidfile'],param['path']['stopwordfile'],False)
        self.__word2vec(param['path']['trainmidfile'],param['path']['wvfile'])
        wv_category_id=self.__get_category_id(param['category'])
        wv_word_id=self.__get_wv_word_id(param['path']['wvfile'],param['path']['wv_wordfile'])
        x_train,y_train=self.__get_XYidtable(param['path']['trainmidfile'],wv_word_id,wv_category_id,param['max_length'])
        x_val,y_val=self.__get_XYidtable(param['path']['valmidfile'],wv_word_id,wv_category_id,param['max_length'])
        wv_vector=__get_wv_vector(wv_word_id,param['path']['wvfile'],param['path']['wv_vectorfile'])

        cnn_train_data=ex_cnn_train_data   
        cnn_train_data['trainXYid_table']=x_train,y_trian
        cnn_train_data['validXY_table']=x_val,y_val
        cnn_train_data['wv_word_size']=len(wv_word_id)
        cnn_train_data['wv_vector_table']=wv_vector
        
        return cnn_train_data
        
    def __cnn_test(self,param):
        cnn_test_data=ex_cnn_test_data
        cnn_test_data['testXYid_table']=self.__get_testXYidtable(param)
        cnn_test_data['wv_word_size']=self.__get_trainvocabsize(param)
        cnn_test_data['wv_vector_table']=self.__get_trainvectable(param)
        
        return cnn_test_data
        
    def __cnn_pred(self,param):
        cnn_pred_data=ex_cnn_pred_data
        cnn_pred_data['predXid_table']=self.__get_predXYidtable(param)
        cnn_pred_data['wv_word_size']=self.__get_trianvocabsize(param)
        cnn_pred_data['wv_vector_table']=self.__get_trainvectable(param)
        
        return cnn_pred_data
        
    def cnn(self,param):
        if param['worktype']==worktype.train:
            dt=self.__cnn_train(param)
        elif param['worktype']==worktype.test:
            dt=self.__cnn_test(param)
        elif param['worktype']==worktype.predict:
            dt=self.__cnn_pred(param)
        else:
            raise
        return dt
            
    def nb(self,param):
        return 1 
    
    def lr(self,param):
        return 2

#########################
#test
#########################
def test_data_cnn():
    dt_config={}
    dt=data(dt_config)
    print(dt.config)
    cnn_param={'worktype':worktype.train}
    cnn_train_data=dt.cnn(cnn_param)
    print(cnn_train_data)
    cnn_param['worktype']=worktype.test
    cnn_test_data=dt.cnn(cnn_param)
    print(cnn_test_data)
    cnn_param['worktype']=worktype.predict
    cnn_pred_data=dt.cnn(cnn_param)
    print(cnn_pred_data)
def test_file2midfile():
    dt_config={}
    dt=data(dt_config)
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\train.txt'
    midpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\midtrain.txt'
    stopwordpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\stopword.txt'    
    dt.file2midfile(path,midpath,stopwordpath,False)
    print('ok')
def test_word2vec():
    dt_config={}
    dt=data(dt_config)
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\4\\train.txt'
    savepath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv.txt'
    dt.word2vec(path,savepath)
def test_get_wv_word():
    dt_config={}
    dt=data(dt_config)
    savepath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_word.txt'
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv.txt'
    dt.wv2word(path,savepath)
def test_get_wv_word_id():
    dt_config={}
    dt=data(dt_config)
    savepath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_word.txt'
    savepath1='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_word1.txt'
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv.txt'
    wv_word_id=dt.get_wv_word_id(path,savepath)
    wv_word_id1=dt.get_wv_word_id(path,savepath1)
    print(wv_word_id==wv_word_id1)
    '''
    for i in wv_word_id:print(i.key+'\t'+i.value)
    print(wv_word_id['优惠'])
    print(len(wv_word_id))'''
def test_wv2vector():
    dt_config={}
    dt=data(dt_config)
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv.txt'
    wvwordpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_word.txt'
    savepath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_vector.npz'
    wv_word_id=dt.get_wv_word_id(wvwordpath)
    dt.wv2vector(wv_word_id,path,savepath)
    print('ok')
def test_get_wv_vector():
    dt_config={}
    dt=data(dt_config)
    wvwordpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_word.txt'
    wvpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv.txt'
    savepath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_vector.npz'
    savepath1='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_vector1.npz'
    wv_word_id=dt.get_wv_word_id(None,wvwordpath)
    vector,vector1=None,None
    vector=dt.get_wv_vector(wv_word_id,wvpath,savepath)
    vector1=dt.get_wv_vector(wv_word_id,wvpath,savepath1)
    print(vector==vector1)
def test_get_XYidtable():
    dt_config={}
    dt=data(dt_config)
    wvwordpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\5\\wv_word.txt'
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\4\\test1.txt'
    wv_word_id=dt.get_wv_word_id(None,wvwordpath)
    category = ['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    wv_category_id=dict(zip(category, range(len(category))))
    dt.get_XYidtable(path,wv_word_id,wv_category_id,100)
if __name__ == '__main__':
    #test_data_cnn()
    #test_file2midfile()
    #test_word2vec()
    #test_get_wv_word()
    #test_get_wv_word_id()
    #test_wv2vector()
    #test_get_wv_vector()
    test_get_XYidtable()
