#encoding:utf-8
import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  

def data_show_pie(labels,num):   
    fig = plt.figure()
    plt.pie(num,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.title("Pie chart")
    plt.show()

if __name__ == '__main__':
    labels=['China','Swiss','USA','UK','Laos','Spain']
    num=[222,42,455,664,454,334]     
    data_show_pie(labels,num)
