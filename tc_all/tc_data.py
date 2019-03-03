#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
@Time   : 2019/3/03
@Author : ZJF
@File   : tc_data.py
'''
import tensorflow.contrib.keras as kr
import jieba.posseg as psg
import numpy as np
import warnings
import logging
import pickle
import jieba
import time
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from gensim.models import word2vec
from collections import  Counter
from datetime import timedelta
from enum import IntEnum
from tc_datatype import *

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

class data(object):
    '''
    '''
    def __init__(self,config):
        '''
        args:
        returns:    
        raises:
        '''
        self.config=config
	
    def  load(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        if param['algorithmtype']==algorithmtype.cnn:
                return self.__cnn(param)
        if param['algorithmtype']==algorithmtype.nb:
                return self.__nb(param)
        if param['algorithmtype']==algorithmtype.lr:
                return self.__lr(param)

###############################	
    def __cnn(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        dt=None
        if param['worktype']==worktype.train:
            dt=self.__cnn_train(param)
        elif param['worktype']==worktype.test:
            dt=self.__cnn_test(param)
        elif param['worktype']==worktype.predict:
            dt=self.__cnn_pred(param)
        else:
            raise
        return dt
            
    def __nb(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        dt=None
        if param['worktype']==worktype.train:
            dt=self.__nb_train(param)
        elif param['worktype']==worktype.test:
            dt=self.__nb_test(param)
        elif param['worktype']==worktype.predict:
            dt=self.__nb_pred(param)
        else:
            raise
        return dt
    
    def __lr(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        return self.__nb(param)

###############################
    def __cnn_train(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        stopword=self.__get_stopword(param['path']['stopwordfile'])
        
        trainfile_string_param,valfile_string_param,stopwordfile_string_param={},{},{}
        trainfile_string_param['file']=param['path']['trainfile']               
        valfile_string_param['file']=param['path']['valfile']
        stopwordfile_string_param['file']=''        
        stopwordfile_string_param['string']=stopword
        
        self.__file2midfile(trainfile_string_param,param['path']['trainmidfile'],stopwordfile_string_param,param['pos']) 
        self.__file2midfile(valfile_string_param,param['path']['valmidfile'],stopwordfile_string_param,param['pos'])
        self.__word2vec(param['path']['trainmidfile'],param['path']['wvfile'])
        wv_category_id=self.__get_category_id(param['category'])
        wv_word_id=self.__get_wv_word_id(param['path']['wvfile'],param['path']['wv_wordfile'])
        trainfile_string_param['file']=param['path']['trainmidfile']
        trainXY=self.__get_XYidtable(trainfile_string_param,wv_word_id,wv_category_id,param['max_length'])
        valXY=self.__get_XYidtable(valfile_string_param,wv_word_id,wv_category_id,param['max_length'])
        wv_vector=self.__get_wv_vector(wv_word_id,param['path']['wvfile'],param['path']['wv_vectorfile'])
        
        cnn_train_data={}   
        cnn_train_data['trainXYid_table']=trainXY
        cnn_train_data['valXYid_table']=valXY
        cnn_train_data['wv_word_size']=len(wv_word_id)
        cnn_train_data['wv_vector_table']=wv_vector
        
        return cnn_train_data
        
    def __cnn_test(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        testfile_string_param,stopwordfile_string_param={},{}
        testfile_string_param['file']=param['path']['testfile']
        stopwordfile_string_param['file']=param['path']['stopwordfile']
        
        self.__file2midfile(testfile_string_param,param['path']['testmidfile'],stopwordfile_string_param,param['pos'])
        wv_category_id=self.__get_category_id(param['category'])
        wv_word_id=self.__get_wv_word_id(param['path']['wvfile'],param['path']['wv_wordfile'])
        testfile_string_param['file']=param['path']['testmidfile']
        x_test,y_test=self.__get_XYidtable(testfile_string_param,wv_word_id,wv_category_id,param['max_length'])
        wv_vector=self.__get_wv_vector(wv_word_id,param['path']['wvfile'],param['path']['wv_vectorfile'])
        
        cnn_test_data={}
        cnn_test_data['testXYid_table']=x_test,y_test
        cnn_test_data['wv_word_size']=len(wv_word_id)
        cnn_test_data['wv_vector_table']=wv_vector
        
        return cnn_test_data
        
    def __cnn_pred(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        predfile_string_param,stopwordfile_string_param={},{}
        stopwordfile_string_param['file']=param['path']['stopwordfile']
        predfile_string_param['file']=param['path']['predfile'] if type(param['path']['predfile'])!=list else ''
        predfile_string_param['string']=param['path']['predfile'] if type(param['path']['predfile'])==list else None
        
        sentence=self.__file2midfile(predfile_string_param,param['path']['predmidfile'],stopwordfile_string_param,param['pos'],pred=True)
        wv_category_id=self.__get_category_id(param['category'])
        wv_word_id=self.__get_wv_word_id(param['path']['wvfile'],param['path']['wv_wordfile'])        
        predfile_string_param['file']=param['path']['predmidfile']
        predfile_string_param['string']=sentence
        x_pred,y_pred=self.__get_XYidtable(predfile_string_param,wv_word_id,wv_category_id,param['max_length'],pred=True)
        wv_vector=self.__get_wv_vector(wv_word_id,param['path']['wvfile'],param['path']['wv_vectorfile'])    
        
        cnn_pred_data={}
        cnn_pred_data['predXid_table']=x_pred,y_pred
        cnn_pred_data['wv_word_size']=len(wv_word_id)
        cnn_pred_data['wv_vector_table']=wv_vector
        
        return cnn_pred_data

    def __nb_train(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        stopword=self.__get_stopword(param['path']['stopwordfile'])
        
        trainfile_string_param,stopwordfile_string_param={},{}
        trainfile_string_param['file']=param['path']['trainfile']
        stopwordfile_string_param['file']=''
        stopwordfile_string_param['string']=stopword
        
        trainbunch=self.__get_midbunch(stopwordfile_string_param,trainfile_string_param,param['path']['trainbunch'],param['category'],param['pos'])
        traintfidfbunch=self.__get_tfidfbunch(stopwordfile_string_param,trainbunch,param['path']['traintfidfbunch'])

        nb_train_data={}
        nb_train_data['traintfidfbunch']=traintfidfbunch
        return nb_train_data
        
    def __nb_test(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        stopword=self.__get_stopword(param['path']['stopwordfile'])
        
        testfile_string_param,stopwordfile_string_param={},{}
        testfile_string_param['file']=param['path']['testfile']
        stopwordfile_string_param['file']=''
        stopwordfile_string_param['string']=stopword
        
        testbunch=self.__get_midbunch(stopwordfile_string_param,testfile_string_param,param['path']['testbunch'],param['category'],param['pos'])
        traintfidfbunch=self.__get_tfidfbunch(None,None,param['path']['traintfidfbunch'])
        testtfidfbunch=self.__get_tfidfbunch(stopwordfile_string_param,testbunch,param['path']['testtfidfbunch'],traintfidfbunch)

        nb_test_data={}
        nb_test_data['testtfidfbunch']=testtfidfbunch
        nb_test_data['traintfidfbunch']=traintfidfbunch
        return nb_test_data

    def __build_nblr_predcontent(self,obj,category):
        '''
        args:
        returns:    
        raises:
        '''
        sentence=self.__gen_sentence(obj) if type(obj)!=list else obj
        predcontent=[]
        for st in sentence:            
            for cat in category:
                tmpstr="{0}\t{1}".format(cat,st)
                predcontent.append(tmpstr)
        return predcontent
    
    def __nb_pred(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        stopword=self.__get_stopword(param['path']['stopwordfile'])
        predcontent=self.__build_nblr_predcontent(param['path']['predfile'],param['category'])
        
        predfile_string_param,stopwordfile_string_param={},{}
        predfile_string_param['file']=''
        predfile_string_param['string']=predcontent
        stopwordfile_string_param['file']=''
        stopwordfile_string_param['string']=stopword        
        
        predbunch=self.__get_midbunch(stopwordfile_string_param,predfile_string_param,param['path']['predbunch'],param['category'],param['pos'])
        traintfidfbunch=self.__get_tfidfbunch(None,None,param['path']['traintfidfbunch'])
        predtfidfbunch=self.__get_tfidfbunch(stopwordfile_string_param,predbunch,param['path']['predtfidfbunch'],traintfidfbunch)

        nb_pred_data={}
        nb_pred_data['predtfidfbunch']=predtfidfbunch
        nb_pred_data['traintfidfbunch']=traintfidfbunch
        return nb_pred_data	
		
###############################
    def __get_wv_word_id(self,wvpath,savepath):
        '''
        args:
        returns:    
        raises:
        '''
        word=None
        if os.path.exists(savepath):
            sentence=self.__gen_sentence(savepath)
            word=[s.strip() for s in sentence]
        else:
            word=self.__wv2word(wvpath,savepath)
        return dict(zip(word, range(len(word))))
        
    def __get_wv_vector(self,wv_word_id,wvpath,savepath):
        '''
        args:
        returns:    
        raises:
        '''
        if os.path.exists(savepath):
            with np.load(savepath) as data:
                return data["embeddings"]
        else:
            return self.__wv2vector(wv_word_id,wvpath,savepath)

    def __get_XYidtable(self,obj,wv_word_id,wv_category_id,max_length,pred=False):
        '''
        args:
        returns:    
        raises:
        '''
        data_id, label_id = [], []
        sentence=self.__gen_sentence(obj['file']) if os.path.isfile(obj['file']) else obj['string']
        for st in sentence:
            if pred:
                data_id.append([wv_word_id[x] for x in st if x in wv_word_id])
            else:
                label,content=st.strip().split('\t')
                data_id.append([wv_word_id[x] for x in content if x in wv_word_id])
                label_id.append(wv_category_id[label.encode('utf-8').decode('utf-8-sig')])
        x = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y = kr.utils.to_categorical(label_id, num_classes=len(wv_category_id))
        return x,y

    def __get_midbunch(self,stopwordobj,midpobj,savebunch,label,pos,savefile=None,pred=False):
        '''
        args:
        returns:    
        raises:
        '''
        bunch=None
        if os.path.exists(savebunch):
            with open(savebunch, "rb") as f:
                bunch = pickle.load(f)
        else:
            bunch=self.__file2midbunch(stopwordobj,midpobj,savebunch,label,pos,savefile,pred)
        return bunch
    
    def __get_tfidfbunch(self,stopwordobj,midbunch,tfidfbunchpath,tfidf_bunch=None):
        '''
        args:
        returns:    
        raises:
        '''
        tfidfbunch=None
        if os.path.exists(tfidfbunchpath):
            with open(tfidfbunchpath, "rb") as f:
                tfidfbunch = pickle.load(f)
        else:
            tfidfbunch=self.__bunch2tfidfbunch(stopwordobj,midbunch,tfidfbunchpath,tfidf_bunch)
        return tfidfbunch

###############################
    def __file2midfile(self,obj,midpath,stopwordobj,pos,pred=False):   
        '''
        args:
        returns:    
        raises:
        '''
        wordsentencelist=[]
        if os.path.exists(midpath):
            return wordsentencelist
        sw=self.__get_stopword(stopwordobj['file']) if os.path.isfile(stopwordobj['file']) else stopwordobj['string'] 
        sentence=self.__gen_sentence(obj['file']) if os.path.isfile(obj['file']) else obj['string']
        for s in sentence:
            fs=self.__format_sentence(s,pred)
            cs=self.__cut_sentence(fs,pos)
            fw=self.__filter_word(cs,sw,pos)
            fws=self.__format_wordsentence(fw,pred)
            if fws:
                wordsentencelist.append(fws)
        if midpath!='':
            with open(midpath,'w',encoding='utf-8') as f:
                f.write('\n'.join(wordsentencelist)+'\n')
        return wordsentencelist
    
    def __word2vec(self,path,savepath,param=None):
        '''
        args:
        returns:    
        raises:
        '''
        if os.path.exists(savepath):
            return
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
        '''
        args:
        returns:    
        raises:
        '''
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
    
    def __wv2vector(self,wv_word_id,wvpath,savepath):
        '''
        args:
        returns:    
        raises:
        '''
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

    def __file2midbunch(self,stopwordobj,obj,savebunch,label,pos,savefile,pred):
        '''
        args:
        returns:    
        raises:
        '''
        bunch=self.__init_bunch(label)
        sw=self.__get_stopword(stopwordobj['file']) if os.path.isfile(stopwordobj['file']) else stopwordobj['string'] 
        sentence=self.__gen_sentence(obj['file']) if os.path.isfile(obj['file']) else obj['string']
        for s in sentence:
            fs=self.__format_sentence(s,pred)
            cs=self.__cut_sentence(fs,pos)
            fw=self.__filter_word(cs,sw,pos)
            cnt=self.__format_wordsentence(fw,pred,tosentence=False)
            if cnt:
                bunch=self.__add_bunch(bunch,cnt[0].replace(',',''),cnt[1]) 
        if savebunch !='':
            with open(savebunch, "wb") as f:
                pickle.dump(bunch, f)
        return bunch
            
    def __bunch2tfidfbunch(self,stopwordobj,bunch,tfidfbunchpath,tfidf_bunch):
        '''
        args:
        returns:    
        raises:
        '''
        sw=self.__get_stopword(stopwordobj['file']) if os.path.isfile(stopwordobj['file']) else stopwordobj['string'] 
        tfidf_space = Bunch(target_name=bunch.target_name, labels=bunch.labels, tdm=[],vocabulary={})
        if tfidf_bunch is not None:
            tfidf_space.vocabulary = tfidf_bunch.vocabulary
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=sw, sublinear_tf=True, max_df=0.8,min_df=0,vocabulary=tfidf_bunch.vocabulary)
            tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
        else:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=sw, sublinear_tf=True, max_df=0.8,min_df=0)
            tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
            tfidf_space.vocabulary = vectorizer.vocabulary_
        if tfidfbunchpath!='':
            with open(tfidfbunchpath, "wb") as f:
                pickle.dump(tfidf_space, f)
        return tfidf_space
		
###############################
    def __get_stopword(self,path):
        '''
        args:
        returns:    
        raises:
        '''
        with open(path, 'r', encoding='utf-8') as f:
            return [sw.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\n', '') for sw in f.readlines()]

    def __gen_sentence(self,path):
        '''
        args:
        returns:    
        raises:
        '''
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

    def __format_sentence(self,sentence,pred):
        '''
        args:
        returns:    
        raises:
        '''
        st=sentence.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\r','').replace('\n', '')
        if pred:
            return st
        return st if '\t' in st else st[:2]+'\t'+st[2:]

    def __cut_sentence(self,sentence,pos):
        '''
        args:
        returns:    
        raises:
        '''
        return psg.cut(sentence) if pos else jieba.cut(sentence)
    
    def __filter_word(self,wordlist,stopwordlist,pos):
        '''
        args:
        returns:    
        raises:
        '''
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
                    if '\t' not in filter_list:
                        filter_list.append(word)
                if not word.isspace():
                    filter_list.append(word)
        return filter_list

    def __format_wordsentence(self,wordlist,pred,tosentence=True):
        '''
        args:
        returns:    
        raises:
        '''
        if len(wordlist) <= 2:
            return None
        if pred:
            return ','.join(wordlist)
        cl,ct=','.join(wordlist).split('\t,')
        if not tosentence:
            return cl,ct
        else:
            return cl.replace(',','')+'\t'+ct

    def __get_category_id(self,category):
        '''
        args:
        returns:    
        raises:
        '''
        return dict(zip(category, range(len(category))))

    def __init_bunch(self,label):
        '''
        args:
        returns:    
        raises:
        '''
        bunch = Bunch(target_name=[], labels=[],  contents=[])
        bunch.target_name.extend(label)
        return bunch

    def __add_bunch(self,bunch,label,content):
        '''
        args:
        returns:    
        raises:
        '''
        bunch.labels.append(label)
        bunch.contents.append(content)
        return bunch
