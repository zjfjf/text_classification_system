#-*- coding:utf-8 -*-
import tkinter as tk
from tc_ui import *

'''
    program start entry
'''

if(__name__=='__main__'):
    root = tk.Tk()
    root.geometry('500x500')
    root.title('Media Player')
    root.minsize(500, 500)
    root.update()
    app = ui(root)
    root.mainloop()
