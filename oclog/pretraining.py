# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:24:31 2022

@author: Bhujay_ROG
"""

import numpy as np
import tensorflow as tf
tf.random.set_seed(123)
from BGL.bglog import  get_embedding_layer
# from bglog import BGLog, get_embedding_layer
# bglog = BGLog(save_padded_num_sequences=False, load_from_pkl=True)
# train_test = bglog.get_tensor_train_test(ablation=1000)
# train_data, test_data = train_test

class LogLineEncoder(tf.keras.Model):
    def __init__(self, logobj, chars_in_line=64, num_of_conv1d=3,  
                 filters=64,
                 kernel_size=3, ):
        super().__init__()  
        self.logobj = logobj
        self.chars_in_line = chars_in_line
        self.num_of_conv1d = num_of_conv1d       
        self.filters = filters
        self.kernel_size = kernel_size
        #TODO Done: make this varaible - bglog - 
        self.embedding_weights, self.vocab_size, self.char_onehot = get_embedding_layer(self.logobj)       
        
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size+1,
                                    output_dim=self.vocab_size,
                                    #TODO: make this varaible
                                    input_length=self.chars_in_line, 
                                    weights = [self.embedding_weights],
                                    )
        self.conv1d_layers = [tf.keras.layers.Conv1D(filters=filters, 
                                                kernel_size=kernel_size, 
                                                padding='same')  
                       for _ in range(self.num_of_conv1d)]
        self.maxpool2d = tf.keras.layers.MaxPooling2D(
            pool_size=(1, self.chars_in_line)) #TODO Done: make this varaible
                  
        
    def call(self, inputs):
        x = self.embedding(inputs)
        for conv1d_layer in self.conv1d_layers:
            x = conv1d_layer(x)
        x = self.maxpool2d(x)
        #TODO: make this varaible
        x = tf.reshape(x, (inputs.shape[0], inputs.shape[1], self.filters))
        return x
    

    
    
class LogSeqEncoder(tf.keras.Model):
    
    def __init__(self, line_in_seq=32, num_of_conv1d=3,  filters=64,
                 kernel_size=3, maxpool_1=True,
                 dense_neurons=16, dense_activation='relu',):
        super().__init__()
        self.line_in_seq = line_in_seq
        self.num_of_conv1d = num_of_conv1d
        self.dense_neurons = dense_neurons
        self.filters = filters
        self.kernel_size = kernel_size
        self.maxpool_1 = maxpool_1
        self.dense_activation = dense_activation
        self.conv1d_layers = [tf.keras.layers.Conv1D(filters=filters, 
                                                kernel_size=kernel_size, 
                                                padding='same')  
                       for _ in range(self.num_of_conv1d)]
        self.maxpool1d = tf.keras.layers.MaxPooling1D(pool_size=(self.line_in_seq) ) #TODO done: make this varaible
        
        self.Dense = tf.keras.layers.Dense(self.dense_neurons, 
                                           activation=self.dense_activation)
       
        
    def call(self, inputs):       
        for conv1d_layer in self.conv1d_layers:
            x = conv1d_layer(inputs)
        x = self.maxpool1d(x)        
        x = tf.reshape(x, (inputs.shape[0], self.filters)) 
        x = self.Dense(x)
        return x
    
    
    
class LogClassifier(tf.keras.Model):
    
    def __init__(self,  num_classes, line_encoder, seq_encoder, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.log_line_encoder = line_encoder
        self.log_seq_encoder = seq_encoder
        self.classifier = tf.keras.layers.Dense(
            self.num_classes, activation='softmax') #TODO done: make this varaible
#         self.extract_feature = extract_feature
    
    def call(self, inputs, extract_feature=False,):
#         x_data, y_data = inputs
        x = self.log_line_encoder(inputs)
        seq_embedding = self.log_seq_encoder(x)
        
        if  extract_feature:
            output = seq_embedding
        else:
            output = self.classifier(seq_embedding)
        return output
    

