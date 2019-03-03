#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
@Time   : 2019/3/03
@Trimmer: ZJF
@File   : tc_nb.py
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tc_data import *
from tc_datatype import *

class nb(object):
    '''
    '''
    def __init__(self,config):
        '''
        args:
        returns:    
        raises:
        '''
        self.config=config

    def run(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        if param['data']['worktype']==worktype.train:
            return self.__do_train(param)
        elif param['data']['worktype']==worktype.test:
            return self.__do_test(param)
        elif param['data']['worktype']==worktype.predict:
            return self.__do_pred(param)
            
    def __do_train(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        traindata=data(None).load(param['data'])
        return ''
    
    def __do_test(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        testdata=data(None).load(param['data'])
        clf = MultinomialNB(alpha=param['nb']['alpha']).fit(testdata['traintfidfbunch'].tdm, testdata['traintfidfbunch'].labels)
        y_true = testdata['testtfidfbunch'].labels
        y_pred = clf.predict(testdata['testtfidfbunch'].tdm)
        report = classification_report(y_true, y_pred, target_names=param['data']['category'])
        mtx = confusion_matrix(y_true, y_pred, labels=param['data']['category'])
        return report,mtx
    
    def __do_pred(self,param):
        '''
        args:
        returns:    
        raises:
        '''
        preddata=data(None).load(param['data'])
        clf = MultinomialNB(alpha=param['nb']['alpha']).fit(preddata['traintfidfbunch'].tdm, preddata['traintfidfbunch'].labels)
        y_pred = clf.predict(preddata['predtfidfbunch'].tdm)
        label=[]
        for i in range(len(y_pred)):
            if(i*8+7>len(y_pred)-1):
                break            
            label.append(Counter(y_pred[i*8:i*8+7]).most_common(1)[0][0])
        return label
