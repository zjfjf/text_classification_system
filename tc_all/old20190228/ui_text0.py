#-*- coding:utf-8 -*-
import tkinter as tk
from tkinter import ttk
import time
import threading

def func_thrd_ExecuteCommand():
    time.sleep(0.01)
    t.delete(0.0,tk.END)
def handle_Input(event):
    if event.keycode==13:          
        global lines
        txt=t.get("0.0", "end")
        tree.insert("",lines,text="" ,values=(txt,lines))        
        tree.yview_moveto(1)
        lines=lines+1
        thrd_once=threading.Thread(target=func_thrd_ExecuteCommand)
        thrd_once.start()

window = tk.Tk()
window.title('my window')
window.geometry('1000x1000')
window.update()

t = tk.Text(window,height=1)
t.bind('<Key>',func=handle_Input)
t.pack(side="top",fill="both")

lines=0
h=window.winfo_height()-t.winfo_height()
tree=ttk.Treeview(window,show="headings",height=h,selectmode='browse')
tree["columns"]=("text","classification")
tree.column("text",width=int(window.winfo_width()*2/3))
tree.column("classification")
tree.heading("text",text="文本",anchor = 'w')
tree.heading("classification",anchor = 'w',text='类别')
vbar = ttk.Scrollbar(window,orient=tk.VERTICAL,command=tree.yview)
vbar.pack(side='right', fill='y')
tree.configure(yscrollcommand=vbar.set)
tree.pack(fill="both",expand="yes")

window.mainloop()
