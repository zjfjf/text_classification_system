# text_classification_system<br>
		application of basic algorithmic model of cnn,nb and lr<br>
##20190301<br>
		学习设计基于循环卷积网络，朴素贝叶斯，逻辑回归算法模型的文本分类系统<br>
###1.参考：<br>
		https://github.com/gaussic/text-classification-cnn-rnn 及所提及资料<br>
		wxid_b8lmlv1vdb2|322提供相关数据及程序素材<br>
###2.大致分为三个模块：数据提供模块，数据处理模块，数据显示模块<br>
		tool：		  工具类			所在文件/模块tc_tool			 提供辅助/扩展方法<br>
		datatype： 	  数据接口类	              所在文件/模块tc_datatype	       提供数据结构，数据协议，规范数据交互<br>
		data：		  数据类			所在文件/模块tc_data			 提供数据<br>
		cnn：		  算法类			所在文件/模块tc_cnn			 卷积神经网络算法<br>
		nb：		  算法类			所在文件/模块tc_nb			 朴素贝叶斯算法<br>
		lr：               算法类			所在文件/模块tc_lr			 逻辑回归算法<br>
		ui：		  界面类			所在文件/模块tc_ui			 提供用户接口<br>
###3.环境：<br>
		Windows10<br>
		Python 3.6.6<br>
		gensim 3.7.1<br>
		jieba 0.39<br>
		numpy 1.16.1<br>
		tensorflow 1.12.0<br>
		matplotlib 3.0.2<br>
###4.后续添加详细说明<br>
###5.其它：<br>
####程序设计过于“简陋”，接下来主要工作：
		* 注意内聚耦合度，可维护性，可扩展性<br>
		* 优化时间空间复杂度，提高性能<br>
		* 提高变量名方法名相关数据接口设计可读性，统一性<br>
		* 提高类及方法体量可观性<br>
		* 提高各个模块间，类间，方法间逻辑清晰度，分配合理性<br>
		* 使用消息队列沟通UI模块与数据模块，实现操作细节显示<br>
		* 使用并发机制，实现多任务独立安全运行<br>
		* 设计实现前后端分离<br>		
		* 重新设计UI结构，寻找/构造高级UI组件,实现复杂数据结构显示，满足数据report可视化需要<br>
		* 调整各算法参数，使其性能最佳<br>
		* 补充完善程序相关说明文档<br>
 ###6.分享&进步<br>
