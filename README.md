# text_classification_system
application basic of algorithmic model of cnn,nb and lr
## 20190301-03
学习设计基于卷积神经网络，朴素贝叶斯，逻辑回归算法模型的文本分类系统
### 1.参考：
https://github.com/gaussic/text-classification-cnn-rnn 及所提及资料<br>	
[wxid_b8lmlv1vdb2|322](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/old20190213) 提供相关数据及程序素材
### 2.大致分为三个模块：数据提供模块，数据处理模块，数据显示模块
|name|type|location|function|
|--|--|--|--|
|tool|工具类|所在文件/模块tc_tool|提供辅助/扩展方法|
|datatype|数据接口类|所在文件/模块tc_datatype|提供数据结构，数据协议，规范数据交互|
|data|数据类|所在文件/模块tc_data|提供数据|
|cnn|算法类|所在文件/模块tc_cnn|卷积神经网络算法|
|nb|算法类|所在文件/模块tc_nb|朴素贝叶斯算法|
|lr|算法类|所在文件/模块tc_lr|逻辑回归算法|
|ui|界面类|所在文件/模块tc_ui|提供用户接口|
|main|入口模块|所在文件/模块tc_main|提供程序入口|
### 3.环境：
* Windows10
* Python 3.6.6
* gensim 3.7.1
* jieba 0.39
* numpy 1.16.1
* tensorflow 1.12.0
* matplotlib 3.0.2 
### 4.功能效果：
* 卷积神经网络<br>
  * 测试(precision-recall-F1-score)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_cnn_test_mtx.png "测试(precision-recall-F1-score)")<br>
  * 测试(confusion matrix)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_cnn_test_report.png "测试(confusion matrix)")<br>
  * 预测(proportion)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_cnn_pred.png "预测(proportion)") <br>
  * 预测(log)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_cnn_pred_log.PNG "预测(log)")<br>	
* 贝叶斯<br>
  * 测试(precision-recall-F1-score)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example//ex_nb_test_mtx.png "测试(precision-recall-F1-score)")<br>
  * 测试(confusion matrix)e<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_nb_test_report.png "测试(confusion matrix)")<br>
  * 预测(proportion)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_nb_pred.png "预测(proportion)")<br>
  * 预测(log)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_nb_pred_log.PNG "预测(log)")<br>	
* 逻辑回归<br>
  * 测试(precision-recall-F1-score)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_lr_test_mtx.png "测试(precision-recall-F1-score)")<br>
  * 测试(confusion matrix)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_lr_test_report.png "测试(confusion matrix)")<br>
  * 预测(proportion)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_lr_pred.png "预测(proportion)")<br>
  * 预测(log)<br>
![](https://github.com/zjfjf/text_classification_system/blob/master/tc_all/data/example/ex_lr_pred_log.PNG "预测(log)")<br>

