#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
@Time   : 2019/3/03
@Author : ZJF
@File   : tc_main.py
'''
import tkinter as tk
from tc_ui import *
'''
    program start entry
'''
if(__name__=='__main__'):
    root = tk.Tk()
    root.geometry('500x500')
    root.title('text classification system')
    root.minsize(500, 500)
    root.update()
    app = ui(root)
    root.mainloop()
