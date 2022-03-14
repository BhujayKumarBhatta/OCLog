# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 07:52:08 2022

@author: Bhujay_ROG
"""

import os
import time
import pickle 
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle


class HDFSLog:
    
    def __init__(self, logpath='C:\ML_data\Logs', 
                 labelpath='C:\ML_data\Logs',
                 logfilename='HDFS.log',
                 labelfilename='anomaly_label.csv',
                 train_ratio=0.8, 
                 split_type='uniform', 
                 save_train_test_data=False):
        self.logpath = logpath
        self.labelpath = labelpath
        self.logfilename = logfilename
        self.labelfilename = labelfilename
        self.logfile = os.path.join(logpath, logfilename)
        self.labelfile = os.path.join(labelpath, labelfilename)
        self.train_ratio = train_ratio
        self.split_type = split_type
        self.save_train_test_data = save_train_test_data
        # self.logs = self.get_log_lines()
        # self.tk = self.train_char_tokenizer()        
        
    def get_train_test_data(self):
        logs = self.get_log_lines()
        tk = self.train_char_tokenizer(logs)
        padded_txt_to_num = self.convert_char_to_numbers(logs, tk)
        sequnce_by_blkids = self.group_logs_by_blkids(logs, padded_txt_to_num)
        labelled_log_sequence = self.label_the_blk_sequences(sequnce_by_blkids)
        x_data = labelled_log_sequence['LogSequence'].values
        y_data = labelled_log_sequence['Label'].values
        x_train, y_train, x_test, y_test = self.train_test_split(x_data, y_data)
        x_train, y_train, x_test, y_test = self.make_equal_len_sequences(x_train, y_train, x_test, y_test)
        return x_train, y_train, x_test, y_test
    
    def get_log_lines(self):
        with open(self.logfile, 'r', encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip().lower() for x in logs]
        n_logs = len(logs)
        print('total number of lines in the log file:', n_logs)
        return logs  
    
    def train_char_tokenizer(self, logs):
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        st_time = time.time()
        print('starting training the tokenizer:')
        tk.fit_on_texts(logs)
        end_time = time.time()
        print('ending tokenizer training:' , end_time - st_time)
        return tk
    
    def convert_char_to_numbers(self, logs, tk):
        print('starting text to number conversion')
        st_time = time.time()
        text_to_number = tk.texts_to_sequences(logs)
        end_time = time.time()
        print('ending text to number conversion:' , end_time - st_time)
        # padded_txt_to_num = pad_sequences(text_to_number, maxlen=256, padding='post')
        # when maxlen is not given, pad_sequences will calculate it automatically
        st_time = time.time()
        padded_txt_to_num = pad_sequences(text_to_number,maxlen=230, padding='post')
        end_time = time.time()
        print('ending padding characters:' , end_time - st_time)
        print('padded_txt_to_num shape:', padded_txt_to_num.shape) # padded_txt_to_num shape: (11175629, 320)
        return padded_txt_to_num
    
    def group_logs_by_blkids(self, logs, padded_txt_to_num):
        data_dict = OrderedDict()
        st_time = time.time()   
        for i, line in enumerate(logs):
           blkId_list = re.findall(r'(blk_-?\d+)', line)
           blkId_list = list(set(blkId_list))
           if len(blkId_list) >=2:
              continue
           blkId_set = set(blkId_list)
           for blk_Id in blkId_set:
             if not blk_Id in data_dict:
                 data_dict[blk_Id] = []
             data_dict[blk_Id].append(padded_txt_to_num[i])
             # if i % 100000: print(blk_Id, data_dict[blk_Id])
             if i % 1000000 == 0: 
                 print('completed: ', i)
                 end_time = time.time()
                 print('ending blk sequencing:' , end_time - st_time)     
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'LogSequence'])
        return data_df
    
    def label_the_blk_sequences(self, data_df):
        label_data = pd.read_csv(self.labelfile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        label_data = '' #release memory
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        return data_df
        
    def train_test_split(self, x_data, y_data=None):
        (x_data, y_data) = shuffle(x_data, y_data)
        if self.split_type == 'uniform' and y_data is not None:
            pos_idx = y_data > 0
            x_pos = x_data[pos_idx]
            y_pos = y_data[pos_idx]
            x_neg = x_data[~pos_idx]
            y_neg = y_data[~pos_idx]
            train_pos = int(self.train_ratio * x_pos.shape[0])
            train_neg = train_pos
            x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
            y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
            x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
            y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        elif self.split_type == 'sequential':
            num_train = int(self.train_ratio * x_data.shape[0])
            x_train = x_data[0:num_train]
            x_test = x_data[num_train:]
            if y_data is None:
                y_train = None
                y_test = None
            else:
                y_train = y_data[0:num_train]
                y_test = y_data[num_train:]
        # Random shuffle
        indexes = shuffle(np.arange(x_train.shape[0]))
        x_train = x_train[indexes]
        if y_train is not None:
            y_train = y_train[indexes]
        print(y_train.sum(), y_test.sum()) # 8419 8419
        return x_train, y_train, x_test, y_test
        
    def make_equal_len_sequences(self, x_train, y_train, x_test, y_test):
        for i in range(100, 120):
            print(len(x_train[i]))            
        padded_x_train = pad_sequences(x_train, maxlen=57, padding='post')  # 57 taken automatically
        ##(16838, 57, 230)        
        for i in range(100, 120):
            print(len(padded_x_train[i]))            
        padded_x_test = pad_sequences(x_test, maxlen=57, padding='post') 
        # # # (558223, 57, 230)
        for i in range(100, 120):
            print(len(padded_x_test[i]))
        if self.save_train_test_data is True:
            with open('data\padded_train.pkl' , 'wb') as f:
                pickle.dump((padded_x_train, y_train), f)
            with open('data\padded_test.pkl' , 'wb') as f:
                pickle.dump((padded_x_test, y_test), f)
        return padded_x_train, y_train, padded_x_test, y_test
    
    
if __name__ == '__main__':
    hdfslog = HDFSLog()
    x_train, y_train, x_test, y_test = hdfslog.get_train_test_data()