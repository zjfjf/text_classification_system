#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import IntEnum

class algorithmtype(IntEnum):
        cnn=0,
        nb=1,
        lr=2,
        pend=3
        
class datatype(IntEnum):
        file=0,
        string=1,
        pend=2

class algorithmtype(IntEnum):
        cnn=0,
        nb=1,
        lr=2,
        pend=3

class worktype(IntEnum):
        train=0,
        test=1,
        predict=2,
        pend=3
	
class predtype(IntEnum):
        file=0,
        string=1,
        pend=2

class cmptype(IntEnum):
        frist=0,
        lase=1,
        total=2,
        pend=3

ex_note_step=['1.\t选择算法，进行step2',
                '2.\t进行训练工作，进行step2.1',
                '2.1.\t点击“...”按钮，选择训练文件train.txt及验证文件val.txt(文本样式：\“IT\t空中上网地面静态测试已完成飞机上可刷微博\”)，进行step2.2',
                '2.2.\t点击“训练”按钮，开始训练，进行step3',
                '3.\t进行测试工作，进行step3.1',
                '3.1.\t点击“...”按钮，选择训练文件test.txt(文本样式：\“IT\t空中上网地面静态测试已完成飞机上可刷微博\”)，进行step3.2',
                '3.2.\t点击“测试”按钮，开始测试，进行step4',
                '4.\t进行预测工作，进行step4.1',
                '4.1.\t点击“...”按钮，选择预测文件pred.txt，或者在输入框中输入预测文本(文本样式：\“空中上网地面静态测试已完成 飞机上可刷微博\”)，进行step4.2',
                '4.2.\t点击“预测”按钮，开始预测，进行step5',
                '5.\t若选择测试，转step3，若进行预测，转step4']

ex_file_name={
                'dir':'',
                'train':'train.txt',
                'trainmid':'trainmid.txt',
                'traindat':'train.dat',
                'traintf':'traintfidf.dat',
                'test':'test.txt',
                'testmid':'testmid.txt',
                'testdat':'test.dat',
                'testtf':'testtfidf.dat',
                'pred':'pred.txt',
                'predmid':'predmid.txt',
                'preddat':'pred.dat',
                'predtf':'predtfidf.dat',
                'val':'val.txt',
                'valmid':'valmid.txt',
                'wv':'wv.txt',
                'word':'wv_word.txt',
                'vector':'wv_vector.txt',
                'best':'best_validation',
                'board':'tensorboard',
                'stop':'stopword.txt'}

ex_test_file_name={
                'dir':'',
                'train':'train1.txt',
                'trainmid':'trainmid1.txt',
                'traindat':'train1.dat',
                'traintf':'traintfidf1.dat',
                'test':'test.txt',
                'testmid':'testmid.txt',
                'testdat':'test.dat',
                'testtf':'testtfidf.dat',
                'pred':'pred.txt',
                'predmid':'predmid.txt',
                'preddat':'pred.dat',
                'predtf':'predtfidf.dat',
                'val':'val.txt',
                'valmid':'valmid.txt',
                'wv':'wv1.txt',
                'word':'wv_word1.txt',
                'vector':'wv_vector1.txt',
                'best':'best_validation1',
                'board':'tensorboard1',
                'stop':'stopword.txt'}

ex_dt_cnn_train_data={
                'trainXYid_table':None,
                'valXYid_table':None,
                'wv_word_size':None,
                'wv_vector_table':None,
                'pend':None}

ex_dt_cnn_test_data={
                'testXYid_table':None,
                'wv_word_size':None,
                'wv_vector_table':None,
                'pend':None}

ex_dt_cnn_pred_data={
                'predXid_table':None,              
                'wv_word_size':None,
                'wv_vector_table':None,
                'pend':None}

ex_dt_data_config={
                'path':None,
                'pos':None,
                'category':None,
                'pend':None}

ex_dt_cnn_train_path={
                'trainfile':None,
                'trainmidfile':None,
                'valfile':None,
                'valmidfile':None,
                'stopwordfile':None,
                'wvfile':None,
                'wv_wordfile':None,
                'wv_vectorfile':None,
                'pend':None}

ex_dt_cnn_test_path={
                'testfile':None,
                'testmidfile':None,
                'stopwordfile':None,
                'wvfile':None,
                'wv_wordfile':None,
                'wv_vectorfile':None,
                'pend':None}

ex_dt_cnn_pred_path={
                'predfile':None,
                'predmidfile':None,
                'stopwordfile':None,
                'wvfile':None,
                'wv_wordfile':None,
                'wv_vectorfile':None,
                'pend':None}

ex_dt_word2vec_param={}

ex_dt_cnn_param={
                'algorithmtype':None,
                'worktype':None,
                'path':None,
                'pos':None,
                'category':None,
                'max_length':None,
                'word2vec':None,
                'pend':None}

ex_dt_file_string_param={
                'file':None,
                'string':None,
                'pend':None}

ex_dt_nb_train_data={
                'traintfidfbunch':None,
                'pend':None}

ex_dt_nb_train_data={
                'traintfidfbunch':None,
                'testtfidfbunch':None,
                'pend':None}

ex_dt_nb_test_data={
                'traintfidfbunch':None,
                'testtfidfbunch':None,
                'pend':None}

ex_dt_nb_train_path={
                'trainfile':None,
                'trainmidfile':None,
                'trainbunch':None,
                'traintfidfbunch':None,
                'stopwordfile':None,
                'pend':None}

ex_dt_nb_test_path={
                'testfile':None,
                'testmidfile':None,
                'testbunch':None,
                'testtfidfbunch':None,
                'traintfidfbunch':None,
                'stopwordfile':None,
                'pend':None}

ex_dt_nb_pred_path={
                'predfile':None,
                'predmidfile':None,
                'predbunch':None,
                'predtfidfbunch':None,
                'traintfidfbunch':None,
                'stopwordfile':None,
                'pend':None}

ex_dt_nb_param={
                'algorithmtype':None,
                'worktype':None,
                'path':None,
                'pos':None,
                'category':None,
                'bunch':None,
                'tfidf':None,
                'pend':None}

ex_dt_lr_train_data={
                'traintfidfbunch':None,
                'pend':None}

ex_dt_lr_test_data={
                'traintfidfbunch':None,
                'testtfidfbunch':None,
                'pend':None}

ex_dt_lr_pred_data={
                'traintfidfbunch':None,
                'predtfidfbunch':None,
                'pend':None}

ex_dt_lr_train_path={
                'trainfile':None,
                'trainmidfile':None,
                'trailrunch':None,
                'traintfidfbunch':None,
                'stopwordfile':None,
                'pend':None}

ex_dt_lr_test_path={
                'testfile':None,
                'testmidfile':None,
                'testbunch':None,
                'testtfidfbunch':None,
                'traintfidfbunch':None,
                'stopwordfile':None,
                'pend':None}

ex_dt_lr_pred_path={
                'predfile':None,
                'predmidfile':None,
                'predbunch':None,
                'predtfidfbunch':None,
                'traintfidfbunch':None,
                'stopwordfile':None,
                'pend':None}

ex_dt_lr_param={
                'algorithmtype':None,
                'worktype':None,
                'path':None,
                'pos':None,
                'category':None,
                'bunch':None,
                'tfidf':None,
                'pend':None}

ex_nb={}

ex_nb_param={
                'data':None,
                'nb':None,
                'pend':None}
ex_lr={}

ex_lr_param={
                'data':None,
                'lr':None,
                'pend':None}

ex_cnn={
                'tensorboarddir':None,
                'savedir':None,
                'savename':None,
                'pend':None}

ex_cnn_param={
                'data':None,
                'cnn':None,
                'pend':None}
ex_cnn_config={
                'embedding_size':200,
                'vocab_size':77216,
                'pre_trianing':None,
                'seq_length':100,
                'num_classes':8,
                'num_filters':128,
                'filter_sizes':[2,3,4],
                'hidden_dim':128,
                'keep_prob':0.5,
                'lr':1e-3,
                'lr_decay':0.9,
                'clip':5.0,
                'num_epochs':10,
                'batch_size':1280,
                'print_per_batch':100,
                'save_per_batch':10}
