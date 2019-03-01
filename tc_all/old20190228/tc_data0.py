#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow.contrib.keras as kr
import numpy as np
import warnings
import codecs
import jieba
import time
import re
import sys
import os
import io
import gc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from gensim.models import word2vec
from collections import  Counter
from datetime import timedelta
from sklearn import metrics

class tc_data(object):

    def __init__(self,sentence_path,stopword_path,batch_size=3,max_length=100,pos=False):
        self.sentence_path=sentence_path
        self.stopword_path=stopword_path
        self.max_length=max_length#
        self.batch_size=batch_size#
        self.pos=pos

        #self.batcher=self.load()
    
    def __get_sentence_list(self):
        return io.open(self.sentence_path,'r',encoding='utf-8').readlines()
    
    def __get_sentence_gen(self):
        with open(self.sentence_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

    def __format_sentence(self,sentence):
        st=sentence.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\r','').replace('\n', '')
        return st if '\t' in st else st[:2]+'\t'+st[2:]

    def __cut_sentence(self,sentence):
        return psg.cut(sentence) if self.pos else jieba.cut(sentence)
    
    def __get_stopword_list(self):
        return [sw.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\n', '') for sw in io.open(self.stopword_path,'r',encoding='utf-8').readlines()]
    
    def __get_stopword_list_w(self):
        with open(self.stopword_path, 'r', encoding='utf-8') as f:
            return [sw.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\n', '') for sw in f.readlines()]
        
    def __filter_word(self,word_list,stopword_list):
        filter_list=[]
        for seg in word_list:
            if not self.pos:
                word = seg
                flag = 'n'
            else:
                word = seg.word
                flag = seg.flag
            if not flag.startswith('n'):
                continue
            if not word in stopword_list:
                if word.isspace() and word == '\t':
                    filter_list.append(word)
                if not word.isspace():
                    filter_list.append(word)
        return filter_list
    
    def __get_wordsentence(self,word_list):
        cl,ct=','.join(word_list).split('\t,')
        return cl.replace(',','')+'\t'+ct
    
    def __get_category_id(self):
        self.categories = ['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
        cat_to_id = dict(zip(self.categories, range(len(self.categories))))
        return cat_to_id
    
    def __get_word_id(self,word_list):
        word_to_id = dict(zip(word_list, range(len(word_list))))
        return word_to_id
    
    def word2vector(self,wordlist_list):
        #model =word2vec.Word2Vec(wordlist_list,size=200, window=5, min_count=2, workers=4)
        #model =word2vec.Word2Vec(wordlist_list,sg=1,hs=1,size=200, window=1, min_count=5, sample=0.001, negative=5, workers=4)
        model =word2vec.Word2Vec(wordlist_list,sg=0,hs=1,size=200, window=1, min_count=5, sample=0.001, negative=5, workers=4)
        #model =word2vec.Word2Vec(wordlist_list,sg=1,hs=1,min_count=5,window=3,size=200, negative=3, sample=0.001,workers=4)
        #model.wv.save_word2vec_format(path, binary=False)                
        model.init_sims(replace=True)
        self.model=model

    def get_content(self):
        label_list=[]
        word_sentence_list=[] 
        
        stop_list=self.__get_stopword_list_w()
        sentence_list=self.__get_sentence_gen()
        for sentence in sentence_list:
            try:
                fmt_sent=self.__format_sentence(sentence)
                word_gen=self.__cut_sentence(fmt_sent)
                word_filter_list=self.__filter_word(word_gen,stop_list)
                label_list.append(word_filter_list[0])
                word_sentence_list.append(word_filter_list[2:])
            except:
                raise
        return label_list,word_sentence_list    

    def word2id(self,model,label_list,word_sentence_list):
        self.cat_iddic=self.__get_category_id()
        self.word_iddic=self.__get_word_id(model.wv.index2word)     
        data_id, label_id = [], []
        
        for i in range(len(word_sentence_list)):
            data_id.append([self.word_iddic[x] for x in word_sentence_list[i] if x in self.word_iddic])
            label_id.append(self.cat_iddic[label_list[i]])
        self.x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.max_length)
        self.y_pad = kr.utils.to_categorical(label_id, num_classes=len(self.cat_iddic))

    def content2id(self,sentence):
        stop_list=self.__get_stopword_list_w()
        fmt_sent=self.__format_sentence(sentence)
        word_gen=self.__cut_sentence(fmt_sent)
        word_filter_list=self.__filter_word(word_gen,stop_list)
        word_sentence_list=[word_filter_list[2:]]    
        data_id=[self.word_iddic[x] for x in word_sentence_list[0] if x in self.word_iddic]     
        return kr.preprocessing.sequence.pad_sequences([data_id], self.max_length)

    def batch(self,x, y):
        data_len = len(x)
        num_batch = int((data_len - 1) / self.batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * self.batch_size
            end_id = min((i + 1) * self.batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def load(self):
        self.label,self.content=self.get_content()
        self.word2vector(self.content)
        self.word2id(self.model,self.label,self.content)
        self.batcher = self.batch(self.x_pad,self.y_pad)
        return self.batcher

if __name__ == "__main__":

    #path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\3\\train.txt'
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\train1\\train_jf.txt'
    stop_path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\stopword.txt'
    
    data=tc_data(path,stop_path)
    data.load()
    print(len(data.content))
    print(len(data.word_iddic))
    print(data.x_pad)
    print(data.y_pad)
    '''
    for x,y in data.batcher:
        print('%s\n%s\n====================='%(x,y))

    print(data.model.wv.vectors.shape)
    '''
