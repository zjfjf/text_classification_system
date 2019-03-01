#-*- coding:utf-8 -*-
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
import tkinter.filedialog as tkf
import tkinter.messagebox as tkm

class ui(tk.Frame):
    
    def __init__(self, master=None):
        self.worktype=worktype.pend
        self.algorithmtype=algorithmtype.pend
        self.filenames={}
        
        super().__init__(master)        
        self.__initcomponent(master)
        
    def __initcomponent(self,master):
        master.rowconfigure(0,weight=1)
        master.columnconfigure(0,weight=1)

        self.text_in = tk.Text(master,height=1)
        self.text_in.bind('<Key>',func=self.__incallback)
        self.text_in.pack(side='top',fill='x')
        self.text_in.insert(0.0,'空中上网地面静态测试已完成 飞机上可刷微博')

        self.frame_cmd=tk.Frame(master)
        self.frame_cmd.pack(side='top',fill='x')
            
        self.btn_file = tk.Button(self.frame_cmd, text ="...", command = self.__filecallback)
        self.btn_file.grid(row=0,column=0)
        self.btn_train = tk.Button(self.frame_cmd, text ="训练", command = lambda :self.__thread_func(self.__traincallback))
        self.btn_train.grid(row=0,column=1)
        self.btn_test = tk.Button(self.frame_cmd, text ="测试", command = lambda :self.__thread_func(self.__testcallback))
        self.btn_test.grid(row=0,column=2)
        self.btn_predict = tk.Button(self.frame_cmd, text ="预测", command = lambda :self.__thread_func(self.__predcallback))
        self.btn_predict.grid(row=0,column=3)
        '''
        self.btn_inpt = tk.Button(self.frame_cmd, text ="中断", command = self.__interruptcallback)
        self.btn_inpt.grid(row=0,column=4)'''
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
            
    def __executedelete(self):
        #time.sleep(0.01)
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
                    'predtfidfbunch':self.filenames['predtf'] if self.__get_txt() =='' else ''
                    #'predtfidfbunch':''
                    }
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
            #thrd=threading.Thread(target=self.__executedelete())
            #thrd.start()
        
    def __clearcallback(self):
        cnt=0
        for i in self.tree.get_children():
            if cnt<10:
                cnt+=1
                continue
            self.tree.delete(i)
        self.__executedelete()
        
    def __interruptcallback(self):
        self.worktype=worktype.pend
        self.__log('','{}--{}'.format(self.algorithmtype,self.worktype),'中断')
        
    def __filecallback(self):
        #self.filenames=ex_file_name
        self.filenames=ex_test_file_name
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
        #threading.Thread(target=self.__runcallback()).start()
        self.__runcallback()
        self.__controlbutton(True)

    def __testcallback(self):
        self.__controlbutton(False)
        self.start=tool.get_time()
        if self.worktype!=worktype.pend:
            self.__log('{}\t正在工作，稍后继续\t...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
            return
        self.worktype=worktype.test
        #threading.Thread(target=self.__runcallback()).start()
        self.__runcallback()
        self.__controlbutton(True)
        
    def __predcallback(self):
        self.__controlbutton(False)
        self.start=tool.get_time()
        if self.worktype!=worktype.pend:
            self.__log('{}\t正在工作，稍后继续...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
            return
        self.worktype=worktype.predict
        #threading.Thread(target=self.__runcallback()).start()
        self.__runcallback()
        self.__controlbutton(True)
        
    def __rbcallback(self):
        self.__log('(\'卷积神经网络\', \'cnn\'), (\'朴素贝叶斯\', \'nb\'), (\'逻辑回归\', \'lr\')',self.foo.get(),'算法')
        self.algorithmtype=algorithmtype[self.foo.get()]
        
###############################
    def __thread_func(self,func, *args):
        t = threading.Thread(target=func, args=args) 
        #t.setDaemon(True)
        t.start()
        
    def __runcallback(self):
        self.__log('{}\t开始...'.format(tool.format_time(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'工作')
        run_param=self.__get_run_param()
        result=None
        if self.algorithmtype==algorithmtype.cnn:
            config=ex_cnn_config
            result=cnn(config).run(run_param)
            print(result)
            #self.__log('{}\t预测：{}'.format(tool.format_time(tool.get_time()),pred),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'完成')
        elif self.algorithmtype==algorithmtype.nb:
            result=nb(None).run(run_param)
            print(result)
            #self.__log('{}\t预测：{}'.format(tool.format_time(tool.get_time()),pred),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'完成')
        elif self.algorithmtype==algorithmtype.lr:
            result=lr(None).run(run_param)
            print(result)
            #self.__log('{}\t预测：{}'.format(tool.format_time(tool.get_time()),pred),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'完成')
        self.__log('{}\t结果：{}'.format(tool.format_time(tool.get_time()),result),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'完成')
        self.__log('{}\t结束,耗时：{}'.format(tool.format_time(tool.get_time()),tool.get_time_dif(self.start)),'{}--{}'.format(self.algorithmtype.name,self.worktype.name),'等待')
        self.worktype=worktype.pend

if(__name__=='__main__'):
    root = tk.Tk()
    root.geometry('500x500')
    root.title('Media Player')
    root.minsize(500, 500)
    root.update()
    app = ui(root)
    root.mainloop()
