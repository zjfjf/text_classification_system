# text_classification_system
application of basic algorithmic model of cnn,nb and lr
#20190301
1.参考：
    https://github.com/gaussic/text-classification-cnn-rnn 及所提及资料
    wxid_b8lmlv1vdb2|322提供相关数据及程序素材
2.大致分为三个模块：数据提供模块，数据处理模块，数据显示模块：
    tool：		  工具类			所在文件/模块tc_tool			 提供辅助/扩展方法
		datatype： 数据接口类	所在文件/模块tc_datatype	 提供数据结构，数据协议，规范数据交互
		data：		  数据类			所在文件/模块tc_data			 提供数据
		cnn：		  算法类			所在文件/模块tc_cnn			   卷积神经网络算法
		nb：			  算法类			所在文件/模块tc_nb			   朴素贝叶斯算法
		lr：       算法类			所在文件/模块tc_lr				 逻辑回归算法
		ui：			  界面类			所在文件/模块tc_ui				 提供用户接口
3.环境：
  Windows10
  Python 3.6.6
  gensim 3.7.1
  jieba 0.39
  numpy 1.16.1
  tensorflow 1.12.0
  matplotlib 3.0.2
4.后续添加详细说明
5.其它：
  程序设计过于“简陋”，接下来主要工作：
		注意内聚耦合度，可维护性，可扩展性
		优化时间空间复杂度，提高性能
		提高变量名方法名相关数据接口设计可读性，统一性
		提高类及方法体量可观性
		提高各个模块间，类间，方法间逻辑清晰度，分配合理性
		使用消息队列沟通UI模块与数据模块，实现操作细节显示
		使用并发机制，实现多任务独立安全运行
		设计实现前后端分离			
		重新设计UI结构，寻找/构造高级UI组件,实现复杂数据结构显示，满足数据report可视化需要
		调整各算法参数，使其性能最佳
    补充完善程序相关说明文档
 6.分享&进步
