#!/usr/bin/python
#encoding:utf-8
"""
@Time   : 2019/1/27
@Author : JYF
@File   : CBAMtest.py
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

slim = tf.contrib.slim

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def convolutional_block_attention_module(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    
	:return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention
        #avg pool
        channel_avg_weights = tf.nn.avg_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        #max pool
        channel_max_weights = tf.nn.max_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        #拼接max pool avg pool
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        #mlp层
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=tf.nn.sigmoid
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        #feature_map经过channel_attention后的输出
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        
        # spatial attention
        #avg pool
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        #max pool
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        #拼接avg pool max pool
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        #卷积，2d->1d
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [3, 3],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        #feature_map经过CBMA模块后输出
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention

if __name__ == '__main__':
    #example
    feature_map = tf.constant(np.random.rand(2,8,8,32), dtype=tf.float16)
    feature_map_with_attention = convolutional_block_attention_module(feature_map,1,0.5)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        result = sess.run(feature_map_with_attention)
        print(result.shape)
