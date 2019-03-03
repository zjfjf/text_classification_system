#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
@Time   : 2019/3/03
@Author : ZJF
@File   : tc_ui.py
'''
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from tc_nb import *
from tc_lr import *
from tc_cnn import *
from tc_tool import *
from tc_data import *
from tc_datatype import *
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
import tkinter.filedialog as tkf
import tkinter.messagebox as tkm
from collections import  Counter
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

class ui(tk.Frame):
    '''
    '''
    def __init__(self, master=None):
        '''
        args:
        returns:    
        raises:
        '''
        super().__init__(master)
        self.__init_component(master)
        self.__init_param()
        self.__config()
        
    def __config(self):
        '''
        args:
        returns:    
        raises:
        '''
        config=tool.get_config()
        if config:
            if config['debug']:
                self.filenames=config['filenames']
                self.worktype_=worktype(config['worktype'])
                self.algorithmtype_=algorithmtype(config['algorithmtype'])
                self.__log('{}\tDEBUG模式...'.format(tool.format_time(tool.get_time())),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'提示')
                for k,v in self.filenames.items():
                    if k=='dir':
                        dir_=v
                    else:
                        self.filenames[k]=dir_+v
                        self.__log(self.filenames[k],os.path.basename(self.filenames[k]),'配置')                 
            self.debug=config['debug']
        else:
            self.debug=False
        
    def __init_param(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.filenames={}
        self.worktype_=worktype.pend
        self.algorithmtype_=algorithmtype.pend

    def __init_component(self,master):
        '''
        args:
        returns:    
        raises:
        '''
        self.__set_root(master)        
        self.__set_inputModule()
        self.__set_cmdModule()
        self.__set_logModule()   
        
###############################
    def __set_root(self,master):
        '''
        args:
        returns:    
        raises:
        '''
        self.master=master
        self.master.rowconfigure(0,weight=1)
        self.master.columnconfigure(0,weight=1)

    def __set_inputModule(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.text_in = tk.Text(self.master,height=1)
        self.text_in.bind('<Key>',func=self.__in_callback)
        self.text_in.pack(side='top',fill='x')

    def __set_cmdModule(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.__set_frame()
        self.__set_button()
        self.__set_rbutton()        
        
    def __set_logModule(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.lines=0
        h=self.master.winfo_height()-self.text_in.winfo_height()-self.frame_cmd.winfo_height()
        
        self.tree=ttk.Treeview(self.master,show="headings",height=h,selectmode='browse')
        self.tree["columns"]=("text","classification","other")
        self.tree.column("text",width=int(self.master.winfo_width()*3/5))
        self.tree.column("classification",width=int(self.master.winfo_width()/5))
        self.tree.column("other",width=int(self.master.winfo_width()/5))
        self.tree.heading("text",text="输出1",anchor = 'w')
        self.tree.heading("classification",anchor = 'w',text='输出2')
        self.tree.heading("other",anchor = 'w',text='输出3')          
        vbar = ttk.Scrollbar(self.master,orient=tk.VERTICAL,command=self.tree.yview)
        vbar.pack(side='right',fill='y')
        self.tree.configure(yscrollcommand=vbar.set)       
        self.tree.pack(side='bottom',fill='both')
        
        note_step=ex_note_step
        for i in note_step:
            self.__log(i.replace('\\',''),i.split('.\t')[0],'步骤')            
        
###############################
    def __set_frame(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.frame_cmd=tk.Frame(self.master)
        self.frame_cmd.pack(side='top',fill='x')

    def __set_button(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.btn_file = tk.Button(self.frame_cmd, text ="...", command = self.__file_callback)
        self.btn_file.grid(row=0,column=0)
        self.btn_train = tk.Button(self.frame_cmd, text ="训练", command = lambda :self.__thread_callback(self.__train_callback))
        self.btn_train.grid(row=0,column=1)
        self.btn_test = tk.Button(self.frame_cmd, text ="测试", command = lambda :self.__thread_callback(self.__test_callback))
        self.btn_test.grid(row=0,column=2)
        self.btn_predict = tk.Button(self.frame_cmd, text ="预测", command = lambda :self.__thread_callback(self.__pred_callback))
        self.btn_predict.grid(row=0,column=3)
        self.btn_clear = tk.Button(self.frame_cmd, text ="清空", command = self.__clear_callback)
        self.btn_clear.grid(row=0,column=4)
        
    def __set_rbutton(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.foo=tk.StringVar(self.master)
        algorithms=ex_algorithms
        clm=5
        for t, v in algorithms:
            rb = tk.Radiobutton(self.frame_cmd, text=t, value=v,variable=self.foo,command=self.__rb_callback)
            rb.grid(row=0,column=clm)
            clm+=1
        self.foo.set('no')
        
###############################
    def __control_button(self,enable):
        '''
        args:
        returns:    
        raises:
        '''
        sate='normal' if enable else 'disable'
        self.btn_train['state']=sate
        self.btn_test['state']=sate
        self.btn_predict['state']=sate
            
    def __delete_text(self,delay=True):
        '''
        args:
        returns:    
        raises:
        '''
        if delay:
            time.sleep(0.01)
        self.text_in.delete(0.0,tk.END)

    def __get_txt(self,last=True):
        '''
        args:
        returns:    
        raises:
        '''
        txt=self.text_in.get("0.0", "end")
        if last:
            return txt if '\n' not in txt else txt.split('\n')[len(txt.split('\n'))-2]
        else:
            return txt

    def __log(self,info1,info2='',info3=''):
        '''
        args:
        returns:    
        raises:
        '''
        tool.log(info1,info2,info3)
        self.tree.insert("",self.lines,text="" ,values=(info1,info2,info3))        
        self.tree.yview_moveto(1)
        self.lines=self.lines+1
        items=self.tree.get_children()
        self.tree.selection_set(items[len(items)-1])
        
    def __get_runparam(self):
        '''
        args:
        returns:    
        raises:
        '''
        txt=self.__get_txt()
        if txt=='':
            pred=self.filenames['pred']
            predmid=self.filenames['predmid']
            preddat=self.filenames['preddat']
            predtf=self.filenames['predtf']
        else:
            pred=[txt]
            self.txt=pred
            predmid=''
            preddat=''
            predtf=''
        category=ex_categoriy_
        savename=os.path.basename(self.filenames['best'])        
        path={
                    'trainfile':self.filenames['train'],
                    'trainmidfile':self.filenames['trainmid'],
                    'valfile':self.filenames['val'],
                    'valmidfile':self.filenames['valmid'],
                    'testfile':self.filenames['test'],
                    'testmidfile':self.filenames['testmid'],
                    'predfile':pred,
                    'predmidfile':predmid,
                    'stopwordfile':self.filenames['stop'],
                    'wvfile':self.filenames['wv'],
                    'wv_wordfile':self.filenames['word'],
                    'wv_vectorfile':self.filenames['vector'],
                    'trainbunch':self.filenames['traindat'],
                    'traintfidfbunch':self.filenames['traintf'],
                    'testbunch':self.filenames['testdat'],
                    'testtfidfbunch':self.filenames['testtf'],                    
                    'predbunch':preddat,
                    'predtfidfbunch':predtf}
        data_param={
                    'algorithmtype':self.algorithmtype_,
                    'worktype':self.worktype_,
                    'path':path,
                    'pos':False,
                    'category':category,
                    'max_length':100}                    
        other_param={                   
                    'savename':savename,
                    'tensorboarddir':self.filenames['board'],
                    'savedir':self.filenames['dir'],
                    'alpha':0.001}
        run_param={
                    'data':data_param,
                    'cnn':other_param,
                    'lr':other_param,
                    'nb':other_param}
        return run_param

###############################
    def __in_callback(self,event):
        '''
        args:
        returns:    
        raises:
        '''
        if event.keycode==13:            
            self.__log(self.__get_txt())
            threading.Thread(target=self.__delete_text()).start()
        
    def __clear_callback(self):
        '''
        args:
        returns:    
        raises:
        '''
        cnt=0
        for i in self.tree.get_children():
            if cnt<10:
                cnt+=1
                continue
            self.tree.delete(i)
        self.__delete_text(False)
        
    def __thread_callback(self,func,*args):
        '''
        args:
        returns:    
        raises:
        '''
        t=threading.Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()

    def __rb_callback(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.__log('(\'卷积神经网络\', \'cnn\'), (\'朴素贝叶斯\', \'nb\'), (\'逻辑回归\', \'lr\')',self.foo.get(),'算法')
        self.algorithmtype_=algorithmtype[self.foo.get()]
        
    def __file_callback(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.filenames=ex_file_name
        files = tkf.askopenfilenames()
        operatefile=''
        if len(files)>0:
            for i in files:
                self.__log(os.path.dirname(i),os.path.basename(i),'选择')
                for j in self.filenames:
                    if j in i:
                        operatefile=os.path.dirname(i)
                        break
        if operatefile != '':
            for k,v in self.filenames.items():
                self.filenames[k]=os.path.join(operatefile,v).replace('/','\\')
                self.__log(self.filenames[k],os.path.basename(self.filenames[k]),'配置') 

###############################
    def __train_callback(self):
        '''
        args:
        returns:    
        raises:
        '''        
        self.start=tool.get_time()
        if not self.__check_input():
            self.__log('{}\参数异常，重新设置\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'警告')
            return
        self.__control_button(False)
        if self.worktype_!=worktype.pend and not self.debug:
            self.__log('{}\t正在工作，稍后继续\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'工作')
            return
        self.worktype_=worktype.train
        self.__run_callback()

    def __test_callback(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.start=tool.get_time()
        if not self.__check_input():
            self.__log('{}\参数异常，重新设置\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'警告')
            return
        self.__control_button(False)
        if self.worktype_!=worktype.pend and not self.debug:
            self.__log('{}\t正在工作，稍后继续\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'工作')
            return
        self.worktype_=worktype.test
        self.__run_callback()
        
    def __pred_callback(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.start=tool.get_time()
        if not self.__check_input():
            self.__log('{}\参数异常，重新设置\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'警告')
            return
        self.__control_button(False)
        if self.worktype_!=worktype.pend and not self.debug:
            self.__log('{}\t正在工作，稍后继续...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'工作')
            return
        self.worktype_=worktype.predict
        self.__run_callback()
        
###############################
    def __check_input(self):
        '''
        args:
        returns:    
        raises:
        '''
        if self.filenames=={} or self.algorithmtype_==algorithmtype.pend:
            return False
        else:
            return True
        
    def __run_callback(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.__log('{}\t开始...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'工作')
        run_param=self.__get_runparam()
        result=None
        if self.algorithmtype_==algorithmtype.cnn:
            config=ex_cnn_config
            result=cnn(config).run(run_param)
        elif self.algorithmtype_==algorithmtype.nb:
            result=nb(None).run(run_param)
        elif self.algorithmtype_==algorithmtype.lr:
            result=lr(None).run(run_param)
        self.__log('{}\t结果：{}'.format(tool.format_time(tool.get_time()),result),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'完成')
        self.__log('{}\t结束,耗时：{}'.format(tool.format_time(tool.get_time()),tool.get_time_dif(self.start)),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'完成')
        self.__data_show(run_param,result)        
        self.worktype_=worktype.pend
        self.__control_button(True)

    def __data_show(self,param,result):
        '''
        args:
        returns:    
        raises:
        '''
        if param['data']['worktype']==worktype.test:
            labelrep,labelmtx=[],[]
            for k in ex_categoriy:
                labelrep.append(k)
                labelmtx.append(k)          
            labelrep.extend(ex_statisticsitem)
            rlt=tool.foramt_reportdata(result[0])
            self.master.bind('<<show>>', lambda evt:self.__data_show_histogram(labelmtx,labelmtx,result[1]))
            self.__thread_callback(self.__triggle_show)
            labelcols=[x for x in ex_evaluatitem]
            self.master.bind('<<show>>', lambda evt:self.__data_show_histogram(labelrep,labelcols,rlt))
            self.__thread_callback(self.__triggle_show)
        elif param['data']['worktype']==worktype.predict:
            length=len(result)            
            self.__data_show_pred(length,result)
            result=Counter(result)
            label,num=zip(*result.most_common(length))
            self.master.bind('<<show>>', lambda evt:self.__data_show_pie(label,num))
            self.__thread_callback(self.__triggle_show)
            
    def __triggle_show(self):
        '''
        args:
        returns:    
        raises:
        '''
        self.master.event_generate('<<show>>', when = "tail")
        
    def __data_show_pred(self,length,pred):
        '''
        args:
        returns:    
        raises:
        '''
        if length>1:
            self.txt= tool.gen_sentence(self.filenames['pred'])
            i=0
            for t in self.txt:
                self.__log('{}\t预测类型：{}--{}'.format(tool.format_time(tool.get_time()),pred[i],t),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'完成')
                i+=1
        else:
            self.__log('{}\t预测类型：{}--{}'.format(tool.format_time(tool.get_time()),pred[0],self.txt[0]),'{}--{}'.format(self.algorithmtype_.name,self.worktype_.name),'完成')        
        
    def __data_show_pie(self,labels,num):
        '''
        args:
        returns:    
        raises:
        '''
        fig = plt.figure()
        plt.pie(num,labels=labels,autopct='%1.2f%%')
        plt.show()

    def __data_show_histogram(self,rows,cols,table):
        '''
        args:
        returns:    
        raises:
        '''
        fig = plt.figure()
        if len(rows)>=10:
            table=[list(x) for x in table]
            table=[x[:3] for x in table]
            X = np.arange(len(cols)-1)+1
        else:
            X = np.arange(len(cols))+1
        aim_color=ex_colors[:len(rows)]        
        ax = fig.add_subplot(111)
        ax.set_xticks(range(len(rows)+1))
        cols.insert(0, '')
        ax.set_xticklabels(cols,rotation=30)        
        bars=[]
        max_=tool.get_max(table)
        division=tool.get_division(max_)
        for i in range(len(table)):
            bar=plt.bar(X+0.05*i,table[i],width = 0.05,facecolor = aim_color[i],edgecolor = 'white')
            bars.append(bar)
            for j in range(len(table[i])):
                plt.text(X[j]+0.05*i, division*i+0.05, '%.2f' % float(table[i][j]), ha='center', va= 'bottom',fontsize=7)
        axlabel=rows if len(rows)>=10 else rows[1:]
        ax.legend(bars,axlabel, fontsize=7)        
        plt.ylim(0,max_+1)
        plt.show()
