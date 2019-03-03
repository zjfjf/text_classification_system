#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@Time   : 2019/3/03
@Author : ZJF
@File   : tc_tool.py
'''
import os
import time
import math
import json
import numpy as np
from tc_datatype import *
from datetime import timedelta

class tool(object):
    '''
    calss for assist
    
    '''
    @classmethod
    def batch_iter(cls,x, y, batch_size=64):
        '''
        distribute data set to several subset
        args:
            x:
                a part of data set that equal input
            y:
                a part of data set that equal result
            size:
                subset length
        returns:
            subset that type is generator            
        raises:
        '''
        data_len=len(x)
        num_batch=int((data_len-1)/batch_size)+1

        indices=np.random.permutation(np.arange(data_len))
        x_shuffle=x[indices]
        y_shuffle=y[indices]

        for i in range(num_batch):
                start_id=i*batch_size
                end_id=min((i+1)*batch_size,data_len)
                yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]
    @classmethod        
    def has_file(cls,_dir,_name,cmp=cmptype.frist):
        '''
        judge directory has aim file or not
        args:
            _dir:
                aim file location
            name:
                aim file
            cmp:
                compare part:
                    frist:
                        prefixion
                    last:
                        postfixion
                    total:
                        all name
                    pend:other
                cmptype from module tc_datatype
        returns:
            subset that type is generator            
        raises:
        '''
        gen=os.walk(_dir)
        obj=[]
        for g in gen:
                obj.append(g)
        for o in obj[0][2]:
                if cmp==cmptype.frist:
                        if os.path.splitext(o)[0]==_name:
                                return True
                elif cmp==cmptype.last:
                        if os.path.splitext(o)[1]==_name:
                                return True
                elif cmp==cmptype.total:
                        if os.path.splitext(o)==_name:
                                return True
        return False
    
    @classmethod        
    def get_time(cls):
        '''
        get current time
        args:
        returns:current time         
        raises:
        '''
        return time.time()
    
    @classmethod
    def format_time(cls,tm):
        '''
        format time display form
        args:
            tm:time ready to format
        returns:
            time has formatted
            example:'2019-03-02 23:10:36'           
        raises:
        '''
        return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tm))
    
    @classmethod         
    def get_time_dif(cls,start_time):
        '''
        calculate space of time
        args:
            start_time:start time
        returns:
            interval of form start time to now      
        raises:
        '''
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    
    @classmethod
    def get_division(cls,num):
        '''
        label form histogram go up distance division avoid overlap
        args:
            num:classify standard.
        returns:
            suitable distance
        raises:
        '''
        num_=len(str(num).split('.')[0])
        if num_<=1:
            return 0.05
        else:
            return math.pow(10,num_ - 2)
        
    @classmethod    
    def get_max(cls,data):
        '''
        get two-dimensional list max value
        args:
            two-dimensional list
        returns:
            max value            
        raises:
        '''
        max_=0.0
        for i in data:
            dl=[float(x) for x in i]
            if max_<max(dl):
                max_=max(dl)
        return max_
    
    @classmethod    
    def foramt_reportdata(cls,report):
        '''
        form report string with letter select number string to data list
        args:
            stirng
        returns:
            data list of float            
        raises:
        '''
        report=report.split('\n')
        report=[list(filter(None, i.split(' '))) for i in report]
        
        report=list(filter(None,report))
        aimnm=[[1,2,3,4,5,6,7,8],[9,10,11]]
        rp=[]
        for i in range(len(report)):
            if i in aimnm[0]:rp.append(report[i][1:])
            elif i in aimnm[1]:rp.append(report[i][2:])
        return [map(float,i) for i in rp]
    
    @classmethod
    def log(cls,*arg):
        '''
        write and format log info to file
        args:
            stirng list
        returns:    
        raises:
        '''
        timestr=str(cls.format_time(cls.get_time()))
        timestr_=str(cls.format_time(cls.get_time()))[:14]
        content=[]
        for item in arg:
            tmp='\n>>'+item
            content.insert(0,tmp)
        content.insert(0,'\n>>'+timestr)
        content.append('\n>>=======================================================================================')        
        logfile=os.path.join('.\\','tc_log\\'+timestr_+'.log')
        logfile=logfile.replace(' ','').replace('/','\\').replace(':','')
        if not os.path.exists(os.path.dirname(logfile)):
            os.makedirs(os.path.dirname(logfile))
        with open(logfile, "a",encoding='utf-8') as fp:
            fp.write(''.join(content).encode('utf-8').decode('utf-8-sig'))
            
    @classmethod
    def gen_sentence(cls,path):
        '''
        read file with lines
        args:
            file path
        returns:
            every lines string and type is generator
        raises:
        '''
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
                
    @classmethod    
    def save_config(cls,dic):
        configfile="./data/config.json"
        if not os.path.exists(configfile):
            os.makedirs(os.path.dirname(configfile))
        with open(configfile,"w",encoding='utf-8') as f:
             json.dump(dic,f)
    @classmethod
    def get_config(cls):
        configfile="./data/config.json"
        if not os.path.exists(configfile):
            return None
        with open(configfile,"r",encoding='utf-8') as f:
             return json.load(f)
            
def test_json():
    tool.save_config(dic = {
        'debug':True,
        'worktype':worktype.train,
        'algorithmtype':algorithmtype,
        'filenames':{
                        'dir':'D:\\DocuemtManagement\\VSproject\\Python_Proj\\Py.Basic\\Text_Categorization_JF\\data_n\\6\\',
                        'train':'train.txt',
                        'trainmid':'trainmid.txt',
                        'traindat':'train.dat',
                        'traintf':'traintfidf.dat',
                        'test':'test.txt',
                        'testmid':'testmid.txt',
                        'testdat':'test.dat',
                        'testtf':'testtfidf.dat',
                        'pred':'pred.txt',
                        'predmid':'predmid.txt',
                        'preddat':'pred.dat',
                        'predtf':'predtfidf.dat',
                        'val':'val.txt',
                        'valmid':'valmid.txt',
                        'wv':'wv.txt',
                        'word':'wv_word.txt',
                        'vector':'wv_vector.txt',
                        'best':'best_validation',
                        'board':'tensorboard',
                        'stop':'stopword.txt'}
                        })

    '''config=tool.get_config()
    for k in config:print(config)
    for k,v in config['filenames'].items():
        if k=='dir':
            dir_=v
        else:
            config['filenames'][k]=dir_+v
    for k in config:print(config)
    tool.save_config(config)'''
    
#test_json()
