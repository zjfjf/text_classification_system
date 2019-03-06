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
### 4.操作步骤：
* cnn训练：
  * 前提：准备文件：
    * train.txt或者trainmid.txt
    * val.txt或者valmid.txt
    * stopword.txt
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为train.txt和val.txt，分别作为训练文件和验证文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为trainmid.txt和valmid.txt，分别作为训练中间文件和验证中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将train.txt/trainmid.txt，val.txt/valmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“循环卷积”
  * 5.点击“...”，进入文件夹A，选择train.txt/trainmid.txt/val.txt/valmid.txt
  * 6.点击训练，开始训练
  * 7.当训练完成，此时文件夹A下有文件：
    * train.txt/trainmid.txt
    * val.txt/valmid.txt
    * stopword.txt
    * wv.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.data-xxxx-of-xxxxx
    * best_validation.index
    * best_validation.meta
    * checkpoint
    * events.out.tfevents.xxxxxxxxxxx.xxxxxxxx-xxxxxxxx
  * 8.备注：若已经有训练好的模型，直接将文件置于文件夹A，进行测试或预测 
    * stopword.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.xxx
    * checkpoint
* cnn测试：
  * 前提：准备文件：
    * test.txt或者testmid.txt
    * stopword.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.xxx
    * checkpoint
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为test.txt，作为测试文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为testmid.txt,作为测试中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将test.txt/testmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“循环卷积”
  * 5.点击“...”，进入文件夹A，选择test.txt/testmid.txt
  * 6.点击测试，开始测试
  * 7.测试完成，此时文件夹A下有文件：
    * test.txt/testmid.txt
    * stopword.txt
    * wv.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.data-xxxx-of-xxxxx
    * best_validation.index
    * best_validation.meta
    * checkpoint
    * events.out.tfevents.xxxxxxxxxxx.xxxxxxxx-xxxxxxxx
* cnn预测（文件）：
  * 前提：准备文件：
    * pred.txt或者predmid.txt
    * stopword.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.xxx
    * checkpoint
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为pred.txt，作为预测文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为predmid.txt,作为预测中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将pred.txt/predmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“循环卷积”
  * 5.点击“...”，进入文件夹A，选择pred.txt/predmid.txt
  * 6.点击预测，开始预测
  * 7.预测完成，此时文件夹A下有文件：
    * pred.txt/predmid.txt
    * stopword.txt
    * wv.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.data-xxxx-of-xxxxx
    * best_validation.index
    * best_validation.meta
    * checkpoint
    * events.out.tfevents.xxxxxxxxxxx.xxxxxxxx-xxxxxxxx
* cnn预测（单文本）：
  * 前提：准备文件：
    * stopword.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.xxx
    * checkpoint
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.软件文本框中输入文本内容样式类似“广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文本
  * 3.2.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.3.将stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“循环卷积”
  * 5.点击预测，开始预测
  * 6.预测完成，此时文件夹A下有文件：
    * stopword.txt
    * wv.txt
    * wv_word.txt
    * wv_vector.txt
    * best_validation.data-xxxx-of-xxxxx
    * best_validation.index
    * best_validation.meta
    * checkpoint
    * events.out.tfevents.xxxxxxxxxxx.xxxxxxxx-xxxxxxxx
* nb训练：
  * 前提：准备文件：
    * train.txt或者trainmid.txt
    * stopword.txt
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为train.txt,作为训练文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为trainmid.txt,作为训练中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将train.txt/trainmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“贝叶斯”
  * 5.点击“...”，进入文件夹A，选择train.txt/trainmid.txt
  * 6.点击训练，开始训练
  * 7.当训练完成，此时文件夹A下有文件：
    * train.txt/trainmid.txt
    * stopword.txt
    * train.dat
    * traintf.dat
  * 8.备注：若已经有训练好的模型，直接将所有文件置于文件夹A，进行测试或预测 
    * train.txt/trainmid.txt
    * stopword.txt
    * train.dat
    * traintf.dat
* nb测试：
  * 前提：准备文件：
    * test.txt或者testmid.txt
    * stopword.txt   
    * traintf.dat
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为test.txt,作为测试文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为testmid.txt,作为测试中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将test.txt/testmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“贝叶斯”
  * 5.点击“...”，进入文件夹A，选择test.txt/testmid.txt
  * 6.点击测试，开始测试
  * 7.当训练完成，此时文件夹A下有文件：
    * test.txt/testmid.txt
    * stopword.txt
    * test.dat
    * testtf.dat
    * traintf.dat
* nb预测（文件）：
  * 前提：准备文件：
    * pred.txt或者predmid.txt
    * stopword.txt
    * traintf.dat
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为pred.txt，作为预测文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为predmid.txt,作为预测中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将pred.txt/predmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“贝叶斯”
  * 5.点击“...”，进入文件夹A，选择pred.txt/predmid.txt
  * 6.点击预测，开始预测
  * 7.预测完成，此时文件夹A下有文件：
    * pred.txt/predmid.txt
    * stopword.txt
    * pred.dat
    * predtf.dat
    * traintf.dat
* nb预测（单文本）：
  * 前提：准备文件：
    * stopword.txt
    * traintf.dat
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.软件文本框中输入文本内容样式类似“广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文本
  * 3.2.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.3.将stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“贝叶斯”
  * 6.点击预测，开始预测
  * 7.预测完成，此时文件夹A下有文件：
    * stopword.txt
    * traintf.dat
* lr训练：
  * 前提：准备文件：
    * train.txt或者trainmid.txt
    * stopword.txt
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为train.txt,作为训练文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为trainmid.txt,作为训练中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将train.txt/trainmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“逻辑回归”
  * 5.点击“...”，进入文件夹A，选择train.txt/trainmid.txt
  * 6.点击训练，开始训练
  * 7.当训练完成，此时文件夹A下有文件：
    * train.txt/trainmid.txt
    * stopword.txt
    * train.dat
    * traintf.dat
  * 8.备注：若已经有训练好的模型，直接将所有文件置于文件夹A，进行测试或预测 
    * train.txt/trainmid.txt
    * stopword.txt
    * train.dat
    * traintf.dat
* lr测试：
  * 前提：准备文件：
    * test.txt或者testmid.txt
    * stopword.txt   
    * traintf.dat
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为test.txt,作为测试文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为testmid.txt,作为测试中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将test.txt/testmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“逻辑回归”
  * 5.点击“...”，进入文件夹A，选择test.txt/testmid.txt
  * 6.点击测试，开始测试
  * 7.当训练完成，此时文件夹A下有文件：
    * test.txt/testmid.txt
    * stopword.txt
    * test.dat
    * testtf.dat
    * traintf.dat
* lr预测（文件）：
  * 前提：准备文件：
    * pred.txt或者predmid.txt
    * stopword.txt
    * traintf.dat
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.是否有文本内容样式类似“IT\t广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文件，若有改名为pred.txt，作为预测文件继续进行步骤3.3，若没有，进行步骤3.2
  * 3.2.是否有文本内容样式类似“IT\t广州，国际，邮件，通关，提速，首推，在线，清关，服务”（样式一致，内容不同）的文件，若有改名为predmid.txt,作为预测中间文件继续进行步骤3.3
  * 3.3.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.4.将pred.txt/predmid.txt，stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“逻辑回归”
  * 5.点击“...”，进入文件夹A，选择pred.txt/predmid.txt
  * 6.点击预测，开始预测
  * 7.预测完成，此时文件夹A下有文件：
    * pred.txt/predmid.txt
    * stopword.txt
    * pred.dat
    * predtf.dat
    * traintf.dat
* lr预测（单文本）：
  * 前提：准备文件：
    * stopword.txt
    * traintf.dat
  * 1.打开config.json文件修改"debug": true ->"debug": False
  * 2.运行python tc_main，出现软件界面
  * 3.1.软件文本框中输入文本内容样式类似“广州国际邮件通关提速首推在线清关服务”（样式一致，内容不同）的文本
  * 3.2.将文本内容样式类似“呢\n吗\n”的文件改名为stopword.txt，作为停顿词文件
  * 3.3.将stopword.txt等文件移到同一文件夹A，比如“C:\tc\”
  * 4.选择算法“逻辑回归”
  * 6.点击预测，开始预测
  * 7.预测完成，此时文件夹A下有文件：
    * stopword.txt
    * traintf.dat
### 5.功能效果：
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
  * 测试(confusion matrix)<br>
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

