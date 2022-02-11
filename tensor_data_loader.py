# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:28:17 2022

@author: Bhujay_ROG
"""
# https://cs230.stanford.edu/blog/datapipeline/
# https://www.tensorflow.org/guide/data
# https://www.tensorflow.org/tutorials/load_data/text
# https://medium.com/deep-learning-with-keras/build-an-efficient-tensorflow-input-pipeline-for-char-level-text-generation-b369d6a68429
# https://www.youtube.com/watch?v=E_kpn3QjGNw

import os
import time
import re
from collections import OrderedDict

import tensorflow as tf
from keras.preprocessing.text import Tokenizer

logpath = os.path.join('C:\ML_data\Logs', 'HDFS.log')
labelpath = os.path.join('C:\ML_data\Logs', 'anomaly_label.csv')

# Read, then decode for py2 compat.
text = open(logpath, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

log_dataset = tf.data.TextLineDataset(logpath)
label_dataset = tf.data.TextLineDataset(labelpath)

# method-0 to iterate , prints the tensor 
# tf.Tensor(b'081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010', shape=(), dtype=string)
i = 0
for elem in log_dataset:
    print(elem)
    i += 1
    if i == 2:
        break
    
# 3 ways of iterating the dataset
data_iterator = iter(log_dataset)

# method-1
next_element = data_iterator.get_next()
for i in range(3):
    print(next_element)

#method-2
nxt = next(data_iterator)
for i in range(3):
    print(nxt.numpy())

#method-3    
for line in log_dataset.take(5):
    print(line.numpy())
    
# method-4 prints the values of the elments
i = 0
for element in log_dataset.as_numpy_iterator():
  print(element)
  i += 1
  if i == 5: break
    
log_dataset.element_spec
# TensorSpec(shape=(), dtype=tf.string, name=None)

# strip and lowe case the log lines
log_dataset = log_dataset.map(lambda x: tf.strings.lower(tf.strings.strip(x)))
for line in log_dataset.take(1):    
    print('lowered and strip:', line.numpy())
    print('lowered and strip:', line)

# pat = r'(blk_-?\d+)'

str = '081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010'
tf.constant(str)

blkid_key = log_dataset.map(lambda x: x[0] )
for line in blkid_key.take(1):    
    print('lowered and strip:', line)  
   
    
    


# split each charecters in the line as token
char_fm_logs = log_dataset.map(lambda x: tf.strings.unicode_split(x, input_encoding='UTF-8'))
for line in char_fm_logs.take(1):
    print(line.numpy())

# character to number converter function 
num_fm_char_func = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

# convert each characters of a line as numbers ir ids 
num_fm_char = char_fm_logs.map(lambda x: num_fm_char_func(x))
for line in num_fm_char.take(1):
    print(line.numpy())

# reverse function - number to character converter
char_fm_num_func = tf.keras.layers.StringLookup(
    vocabulary=num_fm_char_func.get_vocabulary(), invert=True, mask_token=None)

# test the reverse converter have retrived the actual log line from a line of numbers
for line in num_fm_char.take(1):
    # convert each number or ids to its corresponding charecter
    char_fm_num = char_fm_num_func(line)
    print(char_fm_num)
    # join the characters to create the actual log line
    line_fm_chars = tf.strings.reduce_join(char_fm_num, axis=-1)
    print(line_fm_chars)
    


    
    
    

# chars_from_ids = tf.keras.layers.StringLookup(
#     vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    

# tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
# st_time = time.time()
# print('starting training the tokenizer:')
# tk.fit_on_texts(log_dataset)
# end_time = time.time()
# print('ending tokenizer training:' , end_time - st_time)



# window_size = 3
# key_func = lambda line: re.findall(r'(blk_-?\d+)', line)[0] 

# reduce_func = lambda key, 

# line = '081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src:'
# blk = re.findall(r'(blk_-?\d+)', line)
# print(blk)

# for i in range(10):


