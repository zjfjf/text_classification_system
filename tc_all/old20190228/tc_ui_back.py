#-*- coding:utf-8 -*-
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from tc_ui_front import *
import tkinter.filedialog as tkf
import tkinter.messagebox as tkm

def func_thrd_ExecuteCommand():
    time.sleep(0.01)
    text.delete(0.0,tk.END)
def handle_Input(event):
    if event.keycode==13:          
        global lines
        txt=text.get("0.0", "end")
        tree.insert("",lines,text="" ,values=(txt,lines))        
        tree.yview_moveto(1)
        lines=lines+1
        thrd_once=threading.Thread(target=func_thrd_ExecuteCommand)
        thrd_once.start()
def callRB():
    global lines
    tree.insert("",lines,text="" ,values=(foo.get(),'训练','文本'))    
    tree.yview_moveto(1)
    lines=lines+1

def filecallback():
    global lines
    #tkm.showinfo('','请选择配置文件')
    filename = tkf.askopenfilename()
    if filename!='':
        text.insert(0.0,filename)
        tree.insert("",lines,text="" ,values=(filename,'选择文件：',os.path.basename(filename)))
    else:
        tree.insert("",lines,text="" ,values=('选择文件/输入预测文本','无','无'))
    tree.yview_moveto(1)
    lines=lines+1
def traincallback():
    global lines
    tree.insert("",lines,text="" ,values=('训练文本','训练','文本'))    
    tree.yview_moveto(1)
    lines=lines+1
    
    items=tree.get_children()
    tree.selection_set(items[len(items)-1])
def testcallback():
    global lines
    tree.insert("",lines,text="" ,values=('测试文本','文件','文本'))    
    tree.yview_moveto(1)
    lines=lines+1
def predcallback():
    global lines
    tree.insert("",lines,text="" ,values=('预测为本','文件','文本'))    
    tree.yview_moveto(1)
    lines=lines+1
if __name__ == "__main__":
    
    window = tk.Tk()
    window.title('my window')
    window.geometry('500x500')
    window.update()
    window.rowconfigure(0,weight=1)
    window.columnconfigure(0,weight=1)

    text = tk.Text(window,height=1)
    text.bind('<Key>',func=handle_Input)
    text.pack(side='top',fill='x')
    #text.grid(row=1,columnspan=4,sticky='ew')

    frame=tk.Frame(window)
    frame.pack(side='top',fill='x')
    btn_file = tk.Button(frame, text ="...", command = filecallback)
    btn_file.grid(row=0,column=0)
    #btn_file.grid(row=1,column=4,sticky='we')
    btn_train = tk.Button(frame, text ="训练", command = traincallback)
    btn_train.grid(row=0,column=1)
    #btn_train.grid(row=2,column=2,sticky='we')
    btn_test = tk.Button(frame, text ="测试", command = testcallback)
    btn_test.grid(row=0,column=2)
    #btn_test.grid(row=2,column=3,sticky='we')
    btn_predict = tk.Button(frame, text ="预测", command = predcallback)
    btn_predict.grid(row=0,column=3)
    #btn_predict.grid(row=2,column=4,sticky='we')    

    
    #frame.grid(row=2,column=1)
    #frame.rowconfigure(0,weight=1)
    #frame.columnconfigure(0,weight=1)
    foo=tk.IntVar(window)
    i=4
    for t, v in [('卷积神经网络', 1), ('朴素贝叶斯', 2), ('逻辑回归', 3)]:
        r = tk.Radiobutton(frame, text=t, value=v,variable=foo,command=callRB)
        #r.grid(row=0,column=i,sticky='w')
        r.grid(row=0,column=i)
        i+=1
    lines=0
    foo.set(1)
    h=window.winfo_height()-text.winfo_height()-frame.winfo_height()
    tree=ttk.Treeview(window,show="headings",height=h,selectmode='browse')
    tree["columns"]=("text","classification","other")
    tree.column("text",width=int(window.winfo_width()*3/5))
    tree.column("classification",width=int(window.winfo_width()/5))
    tree.column("other",width=int(window.winfo_width()/5))
    tree.heading("text",text="输出1",anchor = 'w')
    tree.heading("classification",anchor = 'w',text='输出2')
    tree.heading("other",anchor = 'w',text='输出3')    
    #tree.grid(row=4,columnspan=4,sticky='nsew')
    vbar = ttk.Scrollbar(window,orient=tk.VERTICAL,command=tree.yview)
    #vbar.grid(row=4,column=4,sticky='ns')
    vbar.pack(side='right',fill='y')
    tree.configure(yscrollcommand=vbar.set)    
    tree.pack(side='bottom',fill='both')
    for j in range(100):
        tree.insert("",lines,text="" ,values=(lines,'文件','文本'))
        lines=lines+1
    window.mainloop()
