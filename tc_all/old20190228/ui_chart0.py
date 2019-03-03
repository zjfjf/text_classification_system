#encoding:utf-8
import numpy as np
import math
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

def show_plot_table():
    data =  [[13,  0,  0,  0,  0,  0,  0,  0],
            [0, 15,  0,  0,  0,  0,  0,  0],
            [ 0,  0, 15,  0,  0,  1,  0,  0],
            [ 0,  0,  0, 26,  0,  0,  0,  0],
            [ 1,  0,  0,  1, 37,  1,  0,  0],
            [ 0,  0,  3,  0,  0, 11,  0,  0],
            [ 0,  0,  0,  0,  0,  0, 17,  0],
            [ 1,  0,  0,  0,  0,  0,  0, 16]]

    columns =['col1','col2','col3','col2','col3','col2','col3','col2']
    rows = ['row1','row2','row3','row1','row2','row3','row1','row2']

    values = np.arange(0,40 ,10)

    # Get some pastel shades for the colors
    colors = ['red','gold','green','blue','yellow','pink','brown','purple']
    n_rows = len(data)

    index = np.arange(len(columns)-1) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row][:7], bar_width, bottom=data[row][:7], color=colors[row])
        #cell_text.append(['%1.1f' % (x) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    #colors = colors[::-1]
    print(cell_text)
    print(rows)
    print(colors)
    print(columns)
    # Add a table at the bottom of the axes
    #colors = colors[::-1]
    #cell_text.reverse()
    the_table = plt.table(cellText=data,
                          rowLabels=rows,
                          rowColours=colors,
                          #cellColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.3)
    
    plt.yticks(values , ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')

    plt.show()

def get_max():
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
                    ['22','23','21','50']]
    max_=''
    for i in table_vals:
        if max_<max(i):
            max_=max(i)
    print(max_)
def get_max_(data):
        max_=0.0
        for i in data:
            dl=[float(x) for x in i]
            if max_<max(dl):
                max_=max(dl)
        return max_
def data_show_plot_table_(row_labels,col_labels,table_vals,rpt=True):
    plt.figure()
    
    data = [['0.87', '1.00', '0.93', '13'], ['1.00', '1.00', '1.00', '15'], ['0.83', '0.94', '0.88', '16'], ['0.96', '1.00', '0.98', '26'], ['1.00', '0.93', '0.96', '40'], ['0.85', '0.79', '0.81', '14'], ['1.00', '1.00', '1.00', '17'], ['1.00', '0.94', '0.97', '17'], ['0.95', '0.95', '0.95', '158'], ['0.94', '0.95', '0.94', '158'], ['0.95', '0.95', '0.95', '158']]
    
    data__=[[87.0, 100.0, 93.0],
             [100.0, 100.0, 100.0],
             [83.0, 94.0, 88.0],
             [96.0, 100.0, 98.0],
             [100.0, 93.0, 96.0],
             [85.0, 79.0, 81.0],
             [100.0, 100.0, 100.0],
             [100.0, 94.0, 97.0],
             [95.0, 95.0, 95.0],
             [94.0, 95.0, 94.0],
             [95.0, 95.0, 95.0]]
    columns =col_labels
    rows = row_labels
    values = np.arange(0,200,100)
    ex_colors=['red','gold','green','blue','yellow','pink','brown','purple','gray','greenyellow','olive']
    row_colors=ex_colors[:len(row_labels)]
    col_colors=ex_colors[:(len(col_labels))]
    n_rows = len(data)
    index = np.arange(len(columns)-1) + 0.3
    bar_width = 0.1*len(col_labels) 
    for row in range(n_rows):
        dt=data__[row]
        plt.bar(index, dt, bar_width, bottom=dt, color=row_colors[row])
    
    the_table = plt.table(cellText=data__,
                          rowLabels=rows,
                          rowColours=row_colors,
                          colLabels=columns[:3],
                          loc='bottom')
    loc_=0.3 if not rpt else 0.4
    plt.subplots_adjust(left=0.2, bottom=0.4)        
    plt.yticks(values , ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')

    plt.show()
#fail color lump distribute unevenness
def show_plot_table__():
    data =  [[87, 100, 93],
     [100, 100, 100],
     [83, 94, 88],
     [96, 100, 98],
     [100, 93, 96],
     [85, 79, 81],
     [100, 100, 100],
     [100, 94, 97],
     [95, 95, 95],
     [94, 95, 94],
     [95, 95, 95]]

    rows =['col1','col2','col3','col2','col3','col2','col3','col2','col2','col3','col2']
    columns = ['row1','row2','row3']

    values = np.arange(0,200 ,100)

    # Get some pastel shades for the colors
    colors = ['red','gold','green','blue','yellow','pink','brown','purple','gray','greenyellow','olive']
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.array([0.0] * len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=data[row], color=colors[row])
        #cell_text.append(['%1.1f' % (x) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    #colors = colors[::-1]
    print(cell_text)
    print(rows)
    print(colors)
    print(columns)
    # Add a table at the bottom of the axes
    #colors = colors[::-1]
    #cell_text.reverse()
    the_table = plt.table(cellText=data,
                          rowLabels=rows,
                          rowColours=colors,
                          #cellColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.4)
    
    plt.yticks(values , ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Loss by Disaster')

    plt.show()
    
def test_mutibar():
    plt.figure(figsize=(9,6))
    n = 8
    X = np.arange(n)+1
    #X是1,2,3,4,5,6,7,8,柱的个数
    # numpy.random.uniform(low=0.0, high=1.0, size=None), normal
    #uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
    Y1 = np.random.uniform(0.5,1.0,n)
    Y2 = np.random.uniform(0.5,1.0,n)
    plt.bar(X,Y1,width = 0.35,facecolor = 'lightskyblue',edgecolor = 'white')
    #width:柱的宽度
    plt.bar(X+0.35,Y2,width = 0.35,facecolor = 'yellowgreen',edgecolor = 'white')
    #水平柱状图plt.barh，属性中宽度width变成了高度height
    #打两组数据时用+
    #facecolor柱状图里填充的颜色
    #edgecolor是边框的颜色
    #想把一组数据打到下边，在数据前使用负号
    #plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
    #给图加text
    for x,y in zip(X,Y1):
        plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
        
    for x,y in zip(X,Y2):
        plt.text(x+0.6, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    plt.ylim(0,+1.25)
    plt.show()
def test_mutibar_(rows,col,table,h):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(col)))
    col.insert(0, '')
    ax.set_xticklabels(col,rotation=30)    
    X = np.arange(len(col)-1)+1 
    colors = ['red','gold','green','blue','yellow','pink','brown','purple','gray','greenyellow','olive']
    aim_color=colors[:len(rows)]
    bars=[]
    for i in range(len(table)):
        bar=plt.bar(X+0.05*i,table[i],width = 0.05,facecolor = aim_color[i],edgecolor = 'white')
        bars.append(bar)
        for j in range(len(table[i])):
            plt.text(X[j]+0.05*i, h*i+0.05, '%.2f' % float(table[i][j]), ha='center', va= 'bottom',fontsize=7)
    ax.legend(bars,rows, fontsize=7)
    max_=get_max_(table)
    plt.ylim(0,max_+1)
    plt.show()

        
if __name__ == '__main__':
    labels=['China','Swiss','USA','UK','Laos','Spain']
    num=[222,42,455,664,454,334]     
    data_show_pie(labels,num)
    
    #data_show_histogram()

    #data_show_table()
    '''table_vals=[
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
    data_show_table_(table_vals)'''

    #show_plot_table__()
    #columns =['col1','col2','col3','col2','col3','col2','col3','col2','col2','col3','col2']
    #data_show_plot_table_(columns[:11],columns[0:4],None)

    #test_mutibar()
    '''colors = ['red','gold','green','blue','yellow','pink','brown','purple','gray','greenyellow','olive']
    data =[[13,  0,  0,  0,  0,  0,  0,  0],
    [0, 15,  0,  0,  0,  0,  0,  0],
    [ 0,  0, 15,  0,  0,  1,  0,  0],
    [ 0,  0,  0, 26,  0,  0,  0,  0],
    [ 1,  0,  0,  1, 37,  1,  0,  0],
    [ 0,  0,  3,  0,  0, 11,  0,  0],
    [ 0,  0,  0,  0,  0,  0, 17,  0],
    [ 1,  0,  0,  0,  0,  0,  0, 16]]

    data =[[0.87, 1.00, 0.93],
     [1.00, 1.00, 1.00],
     [0.83, 0.94, 0.88],
     [0.96, 1.00, 0.98],
     [1.00, 0.93, 0.96],
     [0.85, 0.79, 0.81],
     [1.00, 1.00, 1.00],
     [1.00, 0.94, 0.97],
     [0.95, 0.95, 0.95],
     [0.94, 0.95, 0.94],
     [0.95, 0.95, 0.95]]
    test_mutibar_(colors[:8],colors[:8],data,1)'''
