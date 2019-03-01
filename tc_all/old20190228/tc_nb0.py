#!/usr/bin/python
# -*- coding: utf-8 -*-
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

    def run(self,param):
        if param['data']['worktype']==worktype.train:
            return self.__do_train(param)
        elif param['data']['worktype']==worktype.test:
            return self.__do_test(param)
        elif param['data']['worktype']==worktype.predict:
            return self.__do_pred(param)
            
    def __do_train(self,param):
        dt=data(None)
        traindata=dt.load(param['data'])
        return traindata
    
    def __do_test(self,param):
        dt=data(None)
        testdata=dt.load(param['data'])
        clf = MultinomialNB(alpha=param['alpha']).fit(testdata['traintfidfbunch'].tdm, testdata['traintfidfbunch'].labels)
        y_true = testdata['testtfidfbunch'].labels
        y_pred = clf.predict(testdata['testtfidfbunch'].tdm)
        report = classification_report(y_true, y_pred, target_names=param['data']['category'])
        mtx = confusion_matrix(y_true, y_pred, labels=param['data']['category'])
        return report,mtx
    
    def __do_pred(self,param):
        dt=data(None)
        preddata=dt.load(param['data'])
        clf = MultinomialNB(alpha=param['alpha']).fit(preddata['traintfidfbunch'].tdm, preddata['traintfidfbunch'].labels)
        y_pred = clf.predict(preddata['predtfidfbunch'].tdm)
        print(y_pred)
        label=[]
        for i in range(len(y_pred)):
            if(i*8+7>len(y_pred)-1):
                break            
            label.append(Counter(y_pred[i*8:i*8+7]).most_common(1)[0][0])
        return label      

if __name__ == "__main__":
    nb_train_param={}
    nb_train_path={}
    param={}
    nb_train_path['predfile']=['空中上网地面静态测试已完成 飞机上可刷微博']#'D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\pred.txt'
    nb_train_path['predmidfile']=''
    nb_train_path['predbunch']=''
    nb_train_path['traintfidfbunch']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\traintfidf.dat'
    nb_train_path['predtfidfbunch']=''
    nb_train_path['stopwordfile']='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\stopword.txt'
    nb_train_param['algorithmtype']=algorithmtype.nb
    nb_train_param['worktype']=worktype.predict
    nb_train_param['path']=nb_train_path
    nb_train_param['pos']=False
    nb_train_param['category']=['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融']  
    nb_=nb()
    param['alpha']=0.001
    param['data']=nb_train_param
    pred=nb_.run(param)
    for i in range(len(pred)):
        print(pred[i])
