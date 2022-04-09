# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:24:31 2022

@author: Bhujay_ROG
"""
import numpy as np
import tensorflow as tf
tf.random.set_seed(123)

# def euclidean_metric(a, b):
#     a = np.expand_dims(a, 1)
#     b = np.expand_dims(b, 0)
# #     logits = -((a - b)**2).sum(dim=2)
#     logits = np.sum(-np.square(a - b), axis=2)
#     return logits 

def euclidean_metric(a, b):
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)
#     logits = -((a - b)**2).sum(dim=2)
    logits = tf.math.reduce_sum(-tf.math.square(a - b), axis=2)
    return logits


class BoundaryLoss(tf.keras.layers.Layer):
    def __init__(self, num_labels, 
                feat_dim = 16):
        super().__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        # initializing the delta boundary (4,1 shape is for 4 classes 4 number of scaler value)
        w_init = tf.random_normal_initializer()
        self.theta = tf.Variable(
                            initial_value=w_init(shape=(self.num_labels, 1), dtype='float32'),
                            trainable=True,
                        )
        
    def call(self, features, centroids, labels):
#         ############################
        # delta =  log(1 + e ^ theta) , theta =self.theta = parameters for the boundary
        radius = tf.nn.softplus(self.theta)  
        radius = tf.Variable(radius)
        label_indexs = tf.math.argmax(labels, axis=1)
        c = tf.gather(centroids, indices=label_indexs)
        r = tf.gather(radius, indices=label_indexs)
        x = features
        # x-c = vector of (32, 16) dimension , euc_dis  = scalar value
        euc_dis = tf.norm(x - c, ord='euclidean', axis=1)
        pos_mask = tf.dtypes.cast(euc_dis > r, tf.float32)
        neg_mask = tf.dtypes.cast(euc_dis <= r, tf.float32)
        pos_loss = (euc_dis - r) * pos_mask
        neg_loss = (r - euc_dis) * neg_mask
        loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)        
        return loss, radius
    
    
    