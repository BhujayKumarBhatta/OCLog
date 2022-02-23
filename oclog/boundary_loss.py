# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:24:31 2022

@author: Bhujay_ROG
"""
import numpy as np
import tensorflow as tf
tf.random.set_seed(123)

def euclidean_metric(a, b):
    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 0)
#     logits = -((a - b)**2).sum(dim=2)
    logits = np.sum(-np.square(a - b), axis=2)
    return logits 



class BoundaryLoss(tf.keras.layers.Layer):
    def __init__(self, num_labels, 
                feat_dim = 16):
        super().__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        # initializing the delta boundary (4,1 shape is for 4 classes 4 number of scaler value)
        w_init = tf.random_normal_initializer()
        self.delta = tf.Variable(
                            initial_value=w_init(shape=(self.num_labels, 1), dtype='float32'),
                            trainable=True,
                        )
        
    def call(self, features, centroids, labels):
        logits =  euclidean_metric(features, centroids)  
        ######### Why softmax before softplus#########
        smax = tf.nn.softmax(logits, )
        # this is equivallent to predicting the feature belong to which class
        preds = tf.math.argmax(smax, axis=1)
        # This is equivallent to obtaining the max probabiliy of a feature belonging to a calss
        probs = tf.reduce_max(smax, 1)        
        ############################
        # delta =  log(1 + e ^ delta_k) , delta_k =self.delta = parameters for the boundary
        delta = tf.nn.softplus(self.delta)  
        label_indexs = np.argmax(label_batch, axis=1)
        # centroids are having only 4 rows , whereas labels are rows equivallent to batch
        # pick-up the centroid for each class 
        # label_index from the data set will have all the classes, 32 for a batch
        # for each class cetroid[class_index] will give the centroid of the calss
        # it is basically : [centroids[class_idx] for class_idx in label_indexes]
        c = centroids[label_indexs]
        # similarly get the delta for each class, 
        # although delta is now randomly intialized 
        # delta parameters will be learned through the training
        d = delta[label_indexs]
        x = features
        # x-c = vector of (32, 16) dimension , euc_dis  = scalar value
        euc_dis = tf.norm(x - c, ord='euclidean', axis=1)        
        ##If axis is None (the default), the input is considered a vector and a 
        ## single vector norm is computed over the entire set of values in the tensor, 
        ## i.e. norm(tensor, ord=ord) is equivalent to norm(reshape(tensor, [-1]), ord=ord). 
        ##If axis is a Python integer, the input is considered a batch of vectors, and axis determines the axis in tensor over which to compute vector norms.
        pos_mask = tf.dtypes.cast(euc_dis > d, tf.int32)
        neg_mask = tf.dtypes.cast(euc_dis < d, tf.int32)
        # euc_dis > d should be ==>1 and euc_dis <= d should be ==>0
        # but the expression here will it retrun True , False or 1 and 0. 
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()
        
        return loss, delta
    
    
    