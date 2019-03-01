#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import  Counter
import tensorflow.contrib.keras as kr
import numpy as np
from datetime import timedelta
import time
import os
import sys
from sklearn.feature_selection import SelectKBest
from tc_datatype import *

class tool(object):
	def batch_iter(x, y, batch_size=64):
		data_len=len(x)
		num_batch=int((data_len-1)/batch_size)+1

		indices=np.random.permutation(np.arange(data_len))
		x_shuffle=x[indices]
		y_shuffle=y[indices]

		for i in range(num_batch):
			start_id=i*batch_size
			end_id=min((i+1)*batch_size,data_len)
			yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]
		
	def has_file(_dir,_name,cmp=cmptype.frist):
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
		
	def get_time():
		return time.time()
	
	def format_time(tm):
                return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tm))
                
	def get_time_dif(start_time):
                end_time = time.time()
                time_dif = end_time - start_time
                return timedelta(seconds=int(round(time_dif)))
