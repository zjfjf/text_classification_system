#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import pickle
import time
import sys
import jieba
import jieba.posseg as psg
import io
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import hashlib
from collections import  Counter

stopword_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_ori\\stopword.txt"

train_txt_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\train.txt"
train_dat_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\trainbunch.dat"
train_tfidf_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\traintfidf.dat"

test_txt_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\test_.txt"
test_dat_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\test_.dat"
test_tfidf_path="D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\testtfidf_.dat"

label_list= ['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']

mode = 4

def get_sentence_list(path):
    return io.open(path,'r',encoding='utf-8').readlines()

def format_sentence(sentence):
    st=sentence.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\r','').replace('\n', '')
    return st if '\t' in st else st[:2]+'\t'+st[2:]

def cut_sentence(sentence,pos=False):
    return psg.cut(sentence) if pos else jieba.cut(sentence)

def get_stopword_list(path=stopword_path):
    return [sw.encode('utf-8').decode('utf-8-sig').replace(' ','').replace('\n', '') for sw in io.open(stopword_path,'r',encoding='utf-8').readlines()]

def filter_word(word_list,stopword_list,pos=False):
    filter_list=[]
    for seg in word_list:
        if not pos:
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

def get_wordsentence(word_list):
    cl,ct=','.join(word_list).split('\t,')
    return cl.replace(',','')+'\t'+ct

def init_bunch(label):
    bunch = Bunch(target_name=[], labels=[],  contents=[])
    bunch.target_name.extend(label)
    return bunch

def addcontent_bunch(bunch,label,content):
    bunch.labels.append(label)
    bunch.contents.append(content)
    return bunch

def save_bunch(path,bunch):
    with open(path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)

def load_bunch(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def file2bunch(stop_list,path,pos=False):
    bunch=init_bunch(label_list)
    sentence_list=get_sentence_list(path)
    for sentence in sentence_list:
        try:
            fmt_sent=format_sentence(sentence)
            word_gen=cut_sentence(fmt_sent,pos)
            word_filter_list=filter_word(word_gen,stop_list,pos)
            label,content=','.join(word_filter_list).split('\t,')
            bunch=addcontent_bunch(bunch,label.replace(',',''),content)
        except:
            print(word_filter_list)
            print(','.join(word_filter_list))
            raise
    return bunch

def doc2bunch(stop_list,sentence_list,pos=False):
    bunch=init_bunch(label_list)
    for sentence in sentence_list:
        fmt_sent=format_sentence(sentence)
        word_gen=cut_sentence(fmt_sent,pos)
        word_filter_list=filter_word(word_gen,stop_list,pos)
        label,content=','.join(word_filter_list).split('\t,')
        bunch=addcontent_bunch(bunch,label.replace(',',''),content)
    return bunch

def bunch2tfidf_vector_space(stop_list,bunch,tfidf_bunch=None):
    tfidf_space = Bunch(target_name=bunch.target_name, labels=bunch.labels, tdm=[],vocabulary={})
    if tfidf_bunch is not None:
        tfidf_space.vocabulary = tfidf_bunch.vocabulary
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=stop_list, sublinear_tf=True, max_df=0.8,min_df=0,vocabulary=tfidf_bunch.vocabulary)
        tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
    else:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=stop_list, sublinear_tf=True, max_df=0.8,min_df=0)
        tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
        tfidf_space.vocabulary = vectorizer.vocabulary_
    return tfidf_space

def nb_predict(train_tfidf_space,test_tfidf_space,coef=0.001):
    clf = MultinomialNB(alpha=coef).fit(train_tfidf_space.tdm, train_tfidf_space.labels)
    y_true = test_tfidf_space.labels
    y_pred = clf.predict(test_tfidf_space.tdm)
    report = classification_report(y_true, y_pred, target_names=label_list)
    mtx = confusion_matrix(y_true, y_pred, labels=label_list)
    
    print("预测文本：\n\t{0}\n预测类别：\n\t{1}".format(str,Counter(y_pred).most_common(1)[0][0]))
    return report,mtx

def lr_predict(train_tfidf_space,test_tfidf_space):    
    clf=LogisticRegression(solver='lbfgs', multi_class = 'multinomial', max_iter=50)
    clf.fit(train_tfidf_space.tdm, train_tfidf_space.labels)
    y_true = test_tfidf_space.labels
    y_pred = clf.predict(test_tfidf_space.tdm)
    report = classification_report(y_true, y_pred, target_names=label_list)
    mtx = confusion_matrix(y_true, y_pred, labels=label_list)
    return report,mtx

def svm_predict(train_tfidf_space,test_tfidf_space):
    print("OneVsRestClassifier")
    clf = OneVsRestClassifier(SVC(kernel='linear'))
    print("fit")
    clf.fit(train_tfidf_space.tdm, train_tfidf_space.labels)
    print("true")
    y_true = test_tfidf_space.labels
    print("pred")
    y_pred = clf.predict(test_tfidf_space.tdm)
    print("retport")
    report = classification_report(y_true, y_pred, target_names=label_list)
    print("matrix")
    mtx = confusion_matrix(y_true, y_pred, labels=label_list)
    return report,mtx

def diff_content(A,B):
    #open("D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\1\\train_dat0.dat",'r',encoding='gb18030',errors='ignore').read()
    #open("D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\train_dat0.dat",'r',encoding='gb18030',errors='ignore').read()
    m1 = hashlib.md5()   
    m1.update(a.encode('utf-8'))
    m2 = hashlib.md5()   
    m2.update(b.encode('utf-8'))
    
    print(m1.hexdigest()+'\n'+m2.hexdigest()+'\n'+('true' if m1.hexdigest()==m2.hexdigest() else 'false'))

def foramt_reportdata(report):
    report=report.split('\n')
    report=[list(filter(None, i.split(' '))) for i in report]
    report=list(filter(None,report))
    aimnm=[[1,2,3,4,5,6,7,8],[9,10,11]]
    rp=[]
    for i in range(len(report)):
        if i in aimnm[0]:rp.append(report[i][1:])
        elif i in aimnm[1]:rp.append(report[i][2:])
    return rp
if __name__ == "__main__":
    '''
    str = input("请输入预测文本：")
    print("预测文本：{0}".format(str))'''
    

    test_doc=[]
    for l in label_list:
        tmpstr="{0}\t{1}".format(l,str)
        #print(tmpstr)
        test_doc.append(tmpstr)
        
    
    stop_list=get_stopword_list(stopword_path)
    
    if mode == 0:
        train_bunch=file2bunch(stop_list,train_txt_path)
        train_tfidf_space=bunch2tfidf_vector_space(stop_list,train_bunch)
        test_bunch=file2bunch(stop_list,test_txt_path)
        test_tfidf_space=bunch2tfidf_vector_space(stop_list,test_bunch,train_tfidf_space)
    elif mode == 1:
        train_bunch=file2bunch(stop_list,train_txt_path)
        train_tfidf_space=bunch2tfidf_vector_space(stop_list,train_bunch)
        test_bunch=file2bunch(stop_list,test_txt_path)
        test_tfidf_space=bunch2tfidf_vector_space(stop_list,test_bunch,train_tfidf_space)
        save_bunch(train_dat_path,train_bunch)
        save_bunch(train_tfidf_path,train_tfidf_space)
        #save_bunch(test_dat_path,test_bunch)
        #save_bunch(test_tfidf_path,test_tfidf_space)
    elif mode == 2:
        train_bunch=load_bunch(train_dat_path)
        train_tfidf_space=load_bunch(train_tfidf_path)
        test_bunch=file2bunch(stop_list,test_txt_path)
        test_tfidf_space=bunch2tfidf_vector_space(stop_list,test_bunch,train_tfidf_space)
    elif mode == 3:
        train_bunch=load_bunch(train_dat_path)
        train_tfidf_space=load_bunch(train_tfidf_path)
        test_bunch=doc2bunch(stop_list,test_doc)        
        test_tfidf_space=bunch2tfidf_vector_space(stop_list,test_bunch,train_tfidf_space)
    elif mode == 4:
        train_bunch=load_bunch(train_dat_path)
        train_tfidf_space=load_bunch(train_tfidf_path)
        test_bunch=load_bunch(test_dat_path)
        test_tfidf_space=load_bunch(test_tfidf_path)
              
    #report,mtx=nb_predict(train_tfidf_space,test_tfidf_space)
    #report,mtx=lr_predict(train_tfidf_space,test_tfidf_space)
    report,mtx=nb_predict(train_tfidf_space,test_tfidf_space)
    print(foramt_reportdata(report))
    #print(report)
    #print(mtx)
    '''
    i=0
    index=0
    for m in mtx:
        if m[i]==1:
            index=i
            print("预测文本：\n\t{0}\n预测类别：\n\t{1}".format(str,label_list[i]))
            break
        i=i+1
    '''
