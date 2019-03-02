#-*- coding:utf-8 -*-
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
import math
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

class ui(tk.Frame):
    
    def __init__(self, master=None):        
        super().__init__(master)        
        self.filenames={}
        self.worktype=worktype.pend
        self.algorithmtype=algorithmtype.pend
        self.__initcomponent(master)
        
    def __initcomponent(self,master):
        self.master=master
        master.rowconfigure(0,weight=1)
        master.columnconfigure(0,weight=1)

        self.text_in = tk.Text(master,height=1)
        self.text_in.bind('<Key>',func=self.__incallback)
        self.text_in.pack(side='top',fill='x')

        self.frame_cmd=tk.Frame(master)
        self.frame_cmd.pack(side='top',fill='x')
            
        self.btn_file = tk.Button(self.frame_cmd, text ="...", command = self.__filecallback)
        self.btn_file.grid(row=0,column=0)
        self.btn_train = tk.Button(self.frame_cmd, text ="训练", command = lambda :self.__thread_callback(self.__traincallback))
        self.btn_train.grid(row=0,column=1)
        self.btn_test = tk.Button(self.frame_cmd, text ="测试", command = lambda :self.__thread_callback(self.__testcallback))
        self.btn_test.grid(row=0,column=2)
        self.btn_predict = tk.Button(self.frame_cmd, text ="预测", command = lambda :self.__thread_callback(self.__predcallback))
        self.btn_predict.grid(row=0,column=3)
        self.btn_clear = tk.Button(self.frame_cmd, text ="清空", command = self.__clearcallback)
        self.btn_clear.grid(row=0,column=4)
                
        self.foo=tk.StringVar(master)
        clm=5
        for t, v in [('卷积神经网络', 'cnn'), ('朴素贝叶斯', 'nb'), ('逻辑回归', 'lr'),('None', 'no')]:
            rb = tk.Radiobutton(self.frame_cmd, text=t, value=v,variable=self.foo,command=self.__rbcallback)
            rb.grid(row=0,column=clm)
            clm+=1
        self.foo.set('no')
            
        self.lines=0
        h=master.winfo_height()-self.text_in.winfo_height()-self.frame_cmd.winfo_height()
        self.tree=ttk.Treeview(master,show="headings",height=h,selectmode='browse')
        self.tree["columns"]=("text","classification","other")
        self.tree.column("text",width=int(master.winfo_width()*3/5))
        self.tree.column("classification",width=int(master.winfo_width()/5))
        self.tree.column("other",width=int(master.winfo_width()/5))
        self.tree.heading("text",text="输出1",anchor = 'w')
        self.tree.heading("classification",anchor = 'w',text='输出2')
        self.tree.heading("other",anchor = 'w',text='输出3')          
        vbar = ttk.Scrollbar(master,orient=tk.VERTICAL,command=self.tree.yview)
        vbar.pack(side='right',fill='y')
        self.tree.configure(yscrollcommand=vbar.set)       
        self.tree.pack(side='bottom',fill='both')
        note_step=ex_note_step
        for i in note_step:
            self.__log(i.replace('\\',''),i.split('.\t')[0],'步骤')

###############################
    def __controlbutton(self,enable):
        ctl='normal' if enable else 'disable'
        self.btn_train['state']=ctl
        self.btn_test['state']=ctl
        self.btn_predict['state']=ctl
        
    def __switchbtn1status(self):
        if self.btn_train.instate(['disabled']):
            self.btn_train.state(['!disabled'])
        else:
            self.btn_train.state(['disabled'])
            
    def __deletetext(self,delay=True):
        time.sleep(0.01)
        self.text_in.delete(0.0,tk.END)

    def __log(self,txt,classification='',other=''):
        self.tree.insert("",self.lines,text="" ,values=(txt,classification,other))        
        self.tree.yview_moveto(1)
        self.lines=self.lines+1
        items=self.tree.get_children()
        self.tree.selection_set(items[len(items)-1])
        
    def __get_txt(self,last=True):
        txt=self.text_in.get("0.0", "end")
        if last:
            return txt if '\n' not in txt else txt.split('\n')[len(txt.split('\n'))-2]
        else:
            return txt
        
    def __get_run_param(self):
        path={
                    'trainfile':self.filenames['train'],
                    'trainmidfile':self.filenames['trainmid'],
                    'valfile':self.filenames['val'],
                    'valmidfile':self.filenames['valmid'],
                    'testfile':self.filenames['test'],
                    'testmidfile':self.filenames['testmid'],
                    'predfile':self.filenames['pred'] if self.__get_txt() =='' else [self.__get_txt()],
                    'predmidfile':self.filenames['predmid'] if self.__get_txt() =='' else '',
                    'stopwordfile':self.filenames['stop'],
                    'wvfile':self.filenames['wv'],
                    'wv_wordfile':self.filenames['word'],
                    'wv_vectorfile':self.filenames['vector'],
                    'trainbunch':self.filenames['traindat'],
                    'traintfidfbunch':self.filenames['traintf'],
                    'testbunch':self.filenames['testdat'],
                    'testtfidfbunch':self.filenames['testtf'],                    
                    'predbunch':self.filenames['preddat'] if self.__get_txt() =='' else '',
                    'predtfidfbunch':self.filenames['predtf'] if self.__get_txt() =='' else ''}
        data_param={
                    'algorithmtype':self.algorithmtype,
                    'worktype':self.worktype,
                    'path':path,
                    'pos':False,
                    'category':['IT', '体育', '军事', '娱乐', '文化', '时政', '汽车', '金融'],
                    'max_length':100}                    
        other_param={
                    'tensorboarddir':self.filenames['board'],
                    'savedir':self.filenames['dir'],
                    'savename':os.path.basename(self.filenames['best']),
                    'alpha':0.001}
        run_param={
                    'data':data_param,
                    'cnn':other_param,
                    'lr':other_param,
                    'nb':other_param}
        return run_param

###############################
    def __incallback(self,event):
        if event.keycode==13:            
            self.__log(self.__get_txt())
            threading.Thread(target=self.__deletetext()).start()
        
    def __clearcallback(self):
        cnt=0
        for i in self.tree.get_children():
            if cnt<10:
                cnt+=1
                continue
            self.tree.delete(i)
        self.__deletetext(False)

        
        self.algorithmtype=algorithmtype.nb
        self.filenames=ex_test_file_name
        operatefile='D:\DocuemtManagement\VSproject\Python_Proj\Py.Basic\Text_Categorization_JF\data_n\\6\\'
        if operatefile != '':
            for k,v in self.filenames.items():
                self.filenames[k]=os.path.join(operatefile,v).replace('/','\\')
                self.__log(self.filenames[k],os.path.basename(self.filenames[k]),'配置')
                
    def __interruptcallback(self):
        self.worktype=worktype.pend
        self.__log('','{}--{}'.format(self.algorithmtype,self.worktype),'中断')
        
    def __thread_callback(self,func,*args):
        t=threading.Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()

###############################        
    def __filecallback(self):
        self.filenames=ex_test_file_name#ex_file_name
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
        
    def __traincallback(self):
        self.__controlbutton(False)
        self.start=tool.get_time()
        if self.worktype!=worktype.pend:
            self.__log('{}\t正在工作，稍后继续\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
            return
        self.worktype=worktype.train

    def __testcallback(self):
        self.__controlbutton(False)
        self.start=tool.get_time()
        if self.worktype!=worktype.pend:
            self.__log('{}\t正在工作，稍后继续\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
            return
        self.worktype=worktype.test
        self.__runcallback()
        
    def __predcallback(self):
        self.__controlbutton(False)
        self.start=tool.get_time()
        if self.worktype!=worktype.pend:
            self.__log('{}\t正在工作，稍后继续...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
            return
        self.worktype=worktype.predict
        self.__runcallback()
        
    def __rbcallback(self):
        self.__log('(\'卷积神经网络\', \'cnn\'), (\'朴素贝叶斯\', \'nb\'), (\'逻辑回归\', \'lr\')',self.foo.get(),'算法')
        self.algorithmtype=algorithmtype[self.foo.get()]
        
###############################        
    def __runcallback(self):
        self.__log('{}\t开始...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
        run_param=self.__get_run_param()
        result=None
        if self.algorithmtype==algorithmtype.cnn:
            config=ex_cnn_config
            result=cnn(config).run(run_param)
        elif self.algorithmtype==algorithmtype.nb:
            result=nb(None).run(run_param)
        elif self.algorithmtype==algorithmtype.lr:
            result=lr(None).run(run_param)        
        self.__log('{}\t结果：{}'.format(tool.format_time(tool.get_time()),result),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'完成')
        self.__log('{}\t结束,耗时：{}'.format(tool.format_time(tool.get_time()),tool.get_time_dif(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'等待')
        self.__data_show(run_param,result)        
        self.worktype=worktype.pend
        self.__controlbutton(True)

    def __data_show(self,param,result):
        if param['data']['worktype']==worktype.test:
            labelrep,labelmtx=[],[]
            for k in ex_categoriy:
                labelrep.append(ex_categoriy[k])
                labelmtx.append(ex_categoriy[k])          
            labelrep.extend(ex_statisticsitem)
            rlt=self.__foramt_reportdata(result[0])
            self.master.bind('<<show>>', lambda evt:self.__data_show_histogram(labelmtx,labelmtx,result[1]))
            self.__thread_callback(self.__show)
            self.master.bind('<<show>>', lambda evt:self.__data_show_histogram(labelrep,ex_evaluatitem,rlt))
            self.__thread_callback(self.__show)
        elif param['data']['worktype']==worktype.predict:
            result=Counter(result)
            label,num=zip(*result.most_common(len(result)))
            labels=[ex_categoriy[item] for item in label]
            self.master.bind('<<show>>', lambda evt:self.__data_show_pie(labels,num))
            self.__thread_callback(self.__show)
            
    def __show(self):
        self.master.event_generate('<<show>>', when = "tail")
             
    def __data_show_pie(self,labels,num):
        fig = plt.figure()
        plt.pie(num,labels=labels,autopct='%1.2f%%')
        plt.show()

    def __data_show_histogram(self,rows,cols,table):
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
        max_=self.__get_max(table)
        division=self.__get_division(max_)
        for i in range(len(table)):
            bar=plt.bar(X+0.05*i,table[i],width = 0.05,facecolor = aim_color[i],edgecolor = 'white')
            bars.append(bar)
            for j in range(len(table[i])):
                plt.text(X[j]+0.05*i, division*i+0.05, '%.2f' % float(table[i][j]), ha='center', va= 'bottom',fontsize=7)
        axlabel=rows if len(rows)>=10 else rows[1:]
        ax.legend(bars,axlabel, fontsize=7)        
        plt.ylim(0,max_+1)
        plt.show()

    def __get_division(self,num):
        num_=len(str(num).split('.')[0])
        if num_<=1:
            return 0.05
        else:
            return math.pow(10,num_ - 2)
        
    def __get_max(self,data):
        max_=0.0
        for i in data:
            dl=[float(x) for x in i]
            if max_<max(dl):
                max_=max(dl)
        return max_    
        
    def __foramt_reportdata(self,report):
        report=report.split('\n')
        report=[list(filter(None, i.split(' '))) for i in report]
        report=list(filter(None,report))
        aimnm=[[1,2,3,4,5,6,7,8],[9,10,11]]
        rp=[]
        for i in range(len(report)):
            if i in aimnm[0]:rp.append(report[i][1:])
            elif i in aimnm[1]:rp.append(report[i][2:])
        return [map(float,i) for i in rp]
    
if(__name__=='__main__'):
    root = tk.Tk()
    root.geometry('500x500')
    root.title('Media Player')
    root.minsize(500, 500)
    root.update()
    app = ui(root)
    root.mainloop()
