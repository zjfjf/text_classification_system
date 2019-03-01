#encoding:utf-8
import numpy as np  
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  

def data_show_pie(labels,num):   
    fig = plt.figure()
    plt.pie(num,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.title("Pie chart")
    plt.show()
    
def data_show_histogram():
    np.random.seed(19680801)

    n_bins = 10
    x = np.random.randn(1000, 3)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()

    colors = ['red', 'tan', 'lime']
    ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('bars with legend')

    ax1.hist(x, n_bins, density=True, histtype='barstacked')
    ax1.set_title('stacked bar')

    ax2.hist(x,  histtype='barstacked', rwidth=0.9)

    ax3.hist(x[:, 0], rwidth=0.9)
    ax3.set_title('different sample sizes')

    fig.tight_layout()
    plt.show()

def data_show_table():
    col_labels = ['col1','col2','col3','col2','col3','col2','col3','col2']
    row_labels = ['row1','row2','row3','row1','row2','row3','row1','row2']
    table_vals = [[11,12,13,11,12,13,11,12],
                  [21,22,23,21,22,23,21,22],
                  [28,29,30,21,22,23,21,22],
                  [21,22,23,21,22,23,21,22],
                  [28,29,30,21,22,23,21,22],
                  [21,22,23,21,22,23,21,22],
                  [28,29,30,21,22,23,21,22],
                  [21,22,23,21,22,23,21,22]]
    
    row_colors = ['red','gold','green','blue','yellow','pink','brown','purple']
    my_table = plt.table(cellText=table_vals, colWidths=[0.1]*8,
                         rowLabels=row_labels, colLabels=col_labels,
                         rowColours=row_colors, colColours=row_colors,
                         loc='best')
    my_table.auto_set_font_size(False)
    my_table.set_fontsize(00)

    plt.show()
    
def data_show_table_(table_vals):
    row_colors = ['red','gold','green','blue','yellow','pink','brown','purple','pink','brown','purple']
    col_colors=row_colors[:4]
    my_table = plt.table(cellText=table_vals, colWidths=[0.1]*4,
                         rowLabels=row_colors, colLabels=col_colors,
                         rowColours=row_colors, colColours=col_colors,
                         loc='best')

    plt.show()    

if __name__ == '__main__':
    '''labels=['China','Swiss','USA','UK','Laos','Spain']
    num=[222,42,455,664,454,334]     
    data_show_pie(labels,num)'''
    
    #data_show_histogram()

    #data_show_table()
    table_vals=[
              ['12','13','11','12'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22'],
['22','23','21','22']]
    data_show_table_(table_vals)
