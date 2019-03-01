#!/usr/bin/python
# -*- coding: utf-8 -*-
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
    def __init__(self,config):
        self.config=config
        
#########################
#
#########################
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
                    if '\t' not in filter_list:
                        filter_list.append(word)
                if not word.isspace():
                    filter_list.append(word)
        return filter_list

    def __format_wordsentence(self,wordlist,tosentence=True):
        if len(wordlist) <= 2:
            return None
        cl,ct=','.join(wordlist).split('\t,')
        if not tosentence:
            return cl,ct
        else:
            return cl.replace(',','')+'\t'+ct

    def __get_category_id(self,category):
        return dict(zip(category, range(len(category))))

    def __init_bunch(self,label):
        bunch = Bunch(target_name=[], labels=[],  contents=[])
        bunch.target_name.extend(label)
        return bunch

    def __add_bunch(self,bunch,label,content):
        bunch.labels.append(label)
        bunch.contents.append(content)
        return bunch
        
#########################
#
#########################
    def __file2midfile(self,obj,midpath,stopwordobj,pos):   
        wordsentencelist=[]
        if os.path.exists(midpath):
            return wordsentencelist
        sw=self.__get_stopword(stopwordobj['file']) if os.path.isfile(stopwordobj['file']) else stopwordobj['string'] 
        sentence=self.__gen_sentence(obj['file']) if os.path.isfile(obj['file']) else obj['string']
        for s in sentence:
            fs=self.__format_sentence(s)
            cs=self.__cut_sentence(fs,pos)
            fw=self.__filter_word(cs,sw,pos)
            fws=self.__format_wordsentence(fw)
            if fws:
                wordsentencelist.append(fws)
        if midpath!='':
            with open(midpath,'w',encoding='utf-8') as f:
                f.write('\n'.join(wordsentencelist)+'\n')
        return wordsentencelist
    
    def __word2vec(self,path,savepath,param=None):
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

    def __file2midbunch(self,stopwordobj,obj,savebunch,label,pos,savefile):
        bunch=self.__init_bunch(label)
        sw=self.__get_stopword(stopwordobj['file']) if os.path.isfile(stopwordobj['file']) else stopwordobj['string'] 
        sentence=self.__gen_sentence(obj['file']) if os.path.isfile(obj['file']) else obj['string']
        for s in sentence:
            fs=self.__format_sentence(s)
            cs=self.__cut_sentence(fs,pos)
            fw=self.__filter_word(cs,sw,pos)
            cnt=self.__format_wordsentence(fw,tosentence=False)
            if cnt:
                bunch=self.__add_bunch(bunch,cnt[0].replace(',',''),cnt[1]) 
        if savebunch !='':
            with open(savebunch, "wb") as f:
                pickle.dump(bunch, f)
        return bunch
            
    def __bunch2tfidfbunch(self,stopwordobj,bunch,tfidfbunchpath,tfidf_bunch):
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
#########################
#
#########################
    def __get_wv_word_id(self,wvpath,savepath):
        word=None
        if os.path.exists(savepath):
            sentence=self.__gen_sentence(savepath)
            word=[s.strip() for s in sentence]
        else:
            word=self.__wv2word(wvpath,savepath)
        return dict(zip(word, range(len(word))))
        
    def __get_wv_vector(self,wv_word_id,wvpath,savepath):
        if os.path.exists(savepath):
            with np.load(savepath) as data:
                return data["embeddings"]
        else:
            return self.__wv2vector(wv_word_id,wvpath,savepath)

    def __get_XYidtable(self,obj,wv_word_id,wv_category_id,max_length):
        data_id, label_id = [], []
        sentence=self.__gen_sentence(obj['file']) if os.path.isfile(obj['file']) else obj['string']
        for st in sentence:
            label,content=st.strip().split('\t')
            data_id.append([wv_word_id[x] for x in content if x in wv_word_id])
            label_id.append(wv_category_id[label.encode('utf-8').decode('utf-8-sig')])
        x = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y = kr.utils.to_categorical(label_id, num_classes=len(wv_category_id))
        return x,y

    def __get_midbunch(self,stopwordobj,midpobj,savebunch,label,pos,savefile=None):
        bunch=None
        if os.path.exists(savebunch):
            with open(savebunch, "rb") as f:
                bunch = pickle.load(f)
        else:
            bunch=self.__file2midbunch(stopwordobj,midpobj,savebunch,label,pos,savefile)
        return bunch
    
    def __get_tfidfbunch(self,stopwordobj,midbunch,tfidfbunchpath,tfidf_bunch=None):
        tfidfbunch=None
        if os.path.exists(tfidfbunchpath):
            with open(tfidfbunchpath, "rb") as f:
                tfidfbunch = pickle.load(f)
        else:
            tfidfbunch=self.__bunch2tfidfbunch(stopwordobj,midbunch,tfidfbunchpath,tfidf_bunch)
        return tfidfbunch

#########################
#
#########################
    def __cnn_train(self,param):
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
        testfile_string_param,stopwordfile_string_param={},{}
        testfile_string_param['file']=param['path']['testfile']
        stopwordfile_string_param['file']=param['path']['stopwordfile']
        
        self.__file2midfile(testfile_string_param,param['path']['testmidfile'],stopwordfile_string_param,param['pos'])
        wv_category_id=self.__get_category_id(param['category'])
        wv_word_id=self.__get_wv_word_id(param['path']['wvfile'],param['path']['wv_wordfile'])
        testfile_string_param['file']=param['path']['testmidfile']
        x_test,y_test=self.__get_XYidtable(testfile_string_param,wv_word_id,wv_category_id,param['max_length'])
        wv_vector=self.__get_wv_vector(wv_word_id,param['path']['wvfile'],param['path']['wv_vectorfile'])
        
        cnn_test_data=ex_cnn_test_data
        cnn_test_data['testXYid_table']=x_test,y_test
        cnn_test_data['wv_word_size']=len(wv_word_id)
        cnn_test_data['wv_vector_table']=wv_vector
        
        return cnn_test_data
        
    def __cnn_pred(self,param):
        predfile_string_param,stopwordfile_string_param={},{}
        stopwordfile_string_param['file']=param['path']['stopwordfile']
        predfile_string_param['file']=param['path']['predfile'] if type(param['path']['predfile'])!=list else ''
        predfile_string_param['string']=param['path']['predfile'] if type(param['path']['predfile'])==list else None
        
        sentence=self.__file2midfile(predfile_string_param,param['path']['predmidfile'],stopwordfile_string_param,param['pos'])
        wv_category_id=self.__get_category_id(param['category'])
        wv_word_id=self.__get_wv_word_id(param['path']['wvfile'],param['path']['wv_wordfile'])        
        predfile_string_param['file']=param['path']['predmidfile']
        predfile_string_param['string']=sentence
        x_pred,y_pred=self.__get_XYidtable(predfile_string_param,wv_word_id,wv_category_id,param['max_length'])
        wv_vector=self.__get_wv_vector(wv_word_id,param['path']['wvfile'],param['path']['wv_vectorfile'])    
        
        cnn_pred_data=ex_cnn_pred_data
        cnn_pred_data['predXid_table']=x_pred,y_pred
        cnn_pred_data['wv_word_size']=len(wv_word_id)
        cnn_pred_data['wv_vector_table']=wv_vector
        
        return cnn_pred_data

    def __nb_train(self,param):
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
        sentence=self.__gen_sentence(obj) if type(obj)!=list else obj
        predcontent=[]
        for st in sentence:            
            for cat in category:
                tmpstr="{0}\t{1}".format(cat,st)
                predcontent.append(tmpstr)
        return predcontent
    
    def __nb_pred(self,param):
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
#########################
#
#########################
    def cnn(self,param):
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
            
    def nb(self,param):
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
    
    def lr(self,param):
        return 2

####################################################################################################
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
    path='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\3\\train.txt'
    midpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\midtrain.txt'
    stopwordpath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\stopword.txt'    
    trainfile_string_param={}
    stopwordfile_string_param={}    
    trainfile_string_param['file']=path
    stopwordfile_string_param['file']=stopwordpath
    dt.file2midfile(trainfile_string_param,midpath,stopwordfile_string_param,False)
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
def test_get_wv_vector():
    dt_config={}
    dt=data(dt_config)
    savepath='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'
    vec=dt.get_wv_vector(None,None,savepath)
    '''
    with np.load(savepath) as data1:
        vec=data1["embeddings"]'''
    print('------------')
    print(type(vec))
def test_cnn_train():
    dt_config={}
    dt=data(dt_config)
    cnn_param={}
    cnn_train_path={}
    cnn_train_path['trainfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\train.txt'
    cnn_train_path['trainmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\trainmid.txt'
    cnn_train_path['valfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\val.txt'
    cnn_train_path['valmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\valmid.txt'
    cnn_train_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    cnn_train_path['wvfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt'
    cnn_train_path['wv_wordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt'
    cnn_train_path['wv_vectorfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'
    cnn_param['worktype']=worktype.train
    cnn_param['path']=cnn_train_path
    cnn_param['pos']=False
    cnn_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    cnn_param['max_length']=100
    dt_=dt.cnn(cnn_param)
    print(len(dt_['trainXYid_table'][0]))
    print(len(dt_['trainXYid_table'][1]))
    print(len(dt_['valXYid_table'][0]))
    print(len(dt_['valXYid_table'][1]))
    print(dt_['wv_word_size'])
    print(type(dt_['wv_vector_table']))
def test_cnn_test():
    dt_config={}
    dt=data(dt_config)
    cnn_param={}
    cnn_test_path={}
    cnn_test_path['testfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\test.txt'
    cnn_test_path['testmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\testmid.txt'
    cnn_test_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    cnn_test_path['wvfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt'
    cnn_test_path['wv_wordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt'
    cnn_test_path['wv_vectorfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'
    cnn_param['worktype']=worktype.test
    cnn_param['path']=cnn_test_path
    cnn_param['pos']=False
    cnn_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    cnn_param['max_length']=100
    dt_=dt.cnn(cnn_param)
    print(len(dt_['testXYid_table'][0]))
    print(len(dt_['testXYid_table'][1]))
    print(dt_['wv_word_size'])
    print(len(dt_['wv_vector_table']))
def test_cnn_pred():
    dt_config={}
    dt=data(dt_config)
    cnn_param={}
    cnn_pred_path={}
    cnn_pred_path['predfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\4\\test1.txt'
    cnn_pred_path['predmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\predmid.txt'
    cnn_pred_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    cnn_pred_path['wvfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt'
    cnn_pred_path['wv_wordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt'
    cnn_pred_path['wv_vectorfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'
    cnn_param['worktype']=worktype.predict
    cnn_param['path']=cnn_pred_path
    cnn_param['pos']=False
    cnn_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    cnn_param['max_length']=100
    dt_=dt.cnn(cnn_param)
    print(len(dt_['predXid_table'][0]))
    print(len(dt_['predXid_table'][1]))
    print(dt_['wv_word_size'])
    print(len(dt_['wv_vector_table']))
def test_cnn_pred1():
    dt_config={}
    dt=data(dt_config)
    cnn_param={}
    cnn_pred_path={}
    p='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\4\\test1.txt'
    with open(p,'r',encoding='utf-8') as f:
        st=f.readlines()
    cnn_pred_path['predfile']=st
    #cnn_pred_path['predmidfile']=''
    cnn_pred_path['predmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\predmid1.txt'
    cnn_pred_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    cnn_pred_path['wvfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv.txt'
    cnn_pred_path['wv_wordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_word.txt'
    cnn_pred_path['wv_vectorfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\wv_vector.npz'
    cnn_param['worktype']=worktype.predict
    cnn_param['path']=cnn_pred_path
    cnn_param['pos']=False
    cnn_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    cnn_param['max_length']=100
    dt_=dt.cnn(cnn_param)
    print(len(dt_['predXid_table'][0]))
    print(len(dt_['predXid_table'][1]))
    print(dt_['wv_word_size'])
    print(len(dt_['wv_vector_table']))
def test_nb_train():
    dt_config={}
    dt=data(dt_config)
    nb_param={}
    nb_trian_path={}
    #nb_trian_path['trainfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\4\\test1.txt'
    nb_trian_path['trainfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\train.txt'
    nb_trian_path['trainmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\trainmid.txt'
    nb_trian_path['trainbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\trainbunch.dat'
    nb_trian_path['traintfidfbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\traintfidfbunch.dat'
    nb_trian_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    nb_param['worktype']=worktype.train
    nb_param['path']=nb_trian_path
    nb_param['pos']=False
    nb_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    dt_=dt.nb(nb_param)
    print('OK')
def test_nb_test():
    dt_config={}
    dt=data(dt_config)
    nb_param={}
    nb_test_path={}
    #nb_test_path['testfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\4\\test1.txt'
    nb_test_path['testfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\test.txt'
    nb_test_path['testmidfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\testmid.txt'
    nb_test_path['testbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\testbunch.dat'
    nb_test_path['testtfidfbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\testtfidfbunch.dat'
    nb_test_path['traintfidfbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\traintfidfbunch.dat'
    nb_test_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    nb_param['worktype']=worktype.test
    nb_param['path']=nb_test_path
    nb_param['pos']=False
    nb_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    dt_=dt.nb(nb_param)
    print('OK')   
def test_nb_pred():
    dt_config={}
    dt=data(dt_config)
    nb_param={}
    nb_pred_path={}
    p='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\pred.txt'

    with open(p,'r',encoding='utf-8') as f:
        st=f.readlines()  
    #nb_pred_path['predfile']=p
    nb_pred_path['predfile']=st
    nb_pred_path['predmidfile']=''
    nb_pred_path['predbunch']=''
    nb_pred_path['predtfidfbunch']=''
    nb_pred_path['traintfidfbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\traintfidfbunch.dat'
    nb_pred_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    nb_param['worktype']=worktype.predict
    nb_param['path']=nb_pred_path
    nb_param['pos']=False
    nb_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']
    dt_=dt.nb(nb_param)
    print('OK')    
if __name__ == '__main__':
    #test_data_cnn()
    #test_file2midfile()
    #test_word2vec()
    #test_get_wv_word()
    #test_get_wv_word_id()
    #test_wv2vector()
    #test_get_wv_vector()
    #test_get_XYidtable()
    #test_cnn_train()
    #test_get_wv_vector()
    #test_cnn_test()
    #test_cnn_pred()
    #test_cnn_pred1()
    #test_nb_train()
    #test_nb_test()
    test_nb_pred()
