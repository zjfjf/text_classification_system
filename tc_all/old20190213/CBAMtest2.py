#!/usr/bin/python
#encoding:utf-8
"""
@Time   : 2019/1/28
@Author : JYF
@File   : CBAMtest2.py
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

slim = tf.contrib.slim


def cbam_module(inputs,reduction_ratio=0.5,name=""):

    with tf.variable_scope("cbam_"+name, reuse=tf.AUTO_REUSE):
        batch_size,hidden_num=inputs.get_shape().as_list()[0],inputs.get_shape().as_list()[3]

        maxpool_channel=tf.reduce_max(tf.reduce_max(inputs,axis=1,keepdims=True),axis=2,keepdims=True)
        avgpool_channel=tf.reduce_mean(tf.reduce_mean(inputs,axis=1,keepdims=True),axis=2,keepdims=True)

        
        maxpool_channel = tf.layers.Flatten()(maxpool_channel)
        avgpool_channel = tf.layers.Flatten()(avgpool_channel)

        mlp_1_max=tf.layers.dense(inputs=maxpool_channel,units=int(hidden_num*reduction_ratio),name="mlp_1",reuse=None,activation=tf.nn.relu)
        mlp_2_max=tf.layers.dense(inputs=mlp_1_max,units=hidden_num,name="mlp_2",reuse=None)
        mlp_2_max=tf.reshape(mlp_2_max,[batch_size,1,1,hidden_num])

 
        mlp_1_avg=tf.layers.dense(inputs=avgpool_channel,units=int(hidden_num*reduction_ratio),name="mlp_1",reuse=True,activation=tf.nn.relu)
        mlp_2_avg=tf.layers.dense(inputs=mlp_1_avg,units=hidden_num,name="mlp_2",reuse=True)
        mlp_2_avg=tf.reshape(mlp_2_avg,[batch_size,1,1,hidden_num])

 
        channel_attention=tf.nn.sigmoid(mlp_2_max+mlp_2_avg)
        channel_refined_feature=inputs*channel_attention

 
        maxpool_spatial=tf.reduce_max(inputs,axis=3,keepdims=True)
        avgpool_spatial=tf.reduce_mean(inputs,axis=3,keepdims=True)
        max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
        conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same", activation=None)
        spatial_attention=tf.nn.sigmoid(conv_layer)

        refined_feature=channel_refined_feature*spatial_attention

    return refined_feature

if __name__ == '__main__':
    #example
    feature_map = tf.constant((2,8,8,32), dtype=tf.float16)
    feature_map_with_attention = cbam_module(feature_map, 0.5, "")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        result = sess.run(feature_map_with_attention)
        print(result.shape)
