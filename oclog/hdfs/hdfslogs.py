# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 07:52:08 2022

@author: Bhujay_ROG
"""

import os
import sys
import re
import psutil
import time
import pickle 
import numpy as np
import pandas as pd
from collections import OrderedDict

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# from sklearn.utils import shuffle


class HDFSLog:
    
    def __init__(self, logpath='C:\ML_data\Logs', 
                 labelpath='C:\ML_data\Logs',
                 logfilename='HDFS.log',
                 labelfilename='anomaly_label.csv',
                 train_ratio=0.8, 
                 split_type='uniform', 
                 save_train_test_data=False,
                 padded_seq_len=64,
                 padded_char_len=256,
                 padding_style='post',
                 truncating='pre',
                 rm_time_stamp=True,
                 rm_msg_source=True,
                 rm_blk_ids_regex=True,
                 rm_ip_address=True,
                 rm_signs_n_punctuations=True,
                 rm_white_space=True
                ):
        self.logpath = logpath
        self.labelpath = labelpath
        self.logfilename = logfilename
        self.labelfilename = labelfilename
        self.logfile = os.path.join(logpath, logfilename)
        self.labelfile = os.path.join(labelpath, labelfilename)
        self.train_ratio = train_ratio
        self.split_type = split_type
        self.save_train_test_data = save_train_test_data
        self.padded_seq_len = padded_seq_len
        self.padded_char_len = padded_char_len
        self.padding_style = padding_style
        self.truncating=truncating
        self.rm_time_stamp=True
        self.rm_msg_source=rm_msg_source
        self.rm_blk_ids_regex=rm_blk_ids_regex
        self.rm_ip_address=rm_ip_address
        self.rm_signs_n_punctuations=rm_signs_n_punctuations
        self.rm_white_space=rm_white_space
        self.logs = None
        self.tk = None
        self.padded_txt_to_num = None
        self.seq_of_log_texts = None
        self.seq_of_log_nums = None
        
        
    def get_train_test_data_num(self, ablation=0):
        if self.logs is None:
            self.get_log_lines()
        if self.tk is None:
            self.train_char_tokenizer()
            print('vocabulary size:' , len(self.tk.word_index))
        if self.padded_txt_to_num is None:
            self.convert_char_to_numbers()
        if self.seq_of_log_nums is None:
            self.group_logs_nums_by_blkids()        
        self.label_the_blk_num_seq()               
        x_data = self.seq_of_log_nums['LogSequence'].values
        y_data = self.seq_of_log_nums['Label'].values
        x_train, y_train, x_test, y_test = self.train_test_split(x_data, y_data, ablation=ablation)
        x_train, y_train, x_test, y_test = self.make_equal_len_sequences(x_train, y_train, x_test, y_test)
        print('free ram %: ', psutil.virtual_memory().percent) 
        return x_train, y_train, x_test, y_test
    
    def get_train_test_data_text(self, ablation=0):
        if self.logs is None:
            self.get_log_lines()
        if self.tk is None:
            self.train_char_tokenizer()
            print('vocabulary size:' , len(self.tk.word_index))
        if self.seq_of_log_texts is None:
            self.group_logs_texts_by_blkids()
        self.label_the_blk_txt_seq()             
        x_data = self.seq_of_log_texts['LogSequence'].values
        y_data = self.seq_of_log_texts['Label'].values
        x_train, y_train, x_test, y_test = self.train_test_split(x_data, y_data, ablation=ablation)
        # x_train, y_train, x_test, y_test = self.make_equal_len_sequences(x_train, y_train, x_test, y_test)
        print('free ram %: ', psutil.virtual_memory().percent)
        return x_train, y_train, x_test, y_test
    
    
    def get_log_lines(self):
        with open(self.logfile, 'r', encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip().lower() for x in logs]
        n_logs = len(logs)
        print('total number of lines in the log file:', n_logs)
        print('RAM usage: ', sys.getsizeof(logs) )
        self.logs = logs
        return logs  
    
    def remove_unwanted_characters_n_words(self, txt_line, debug=False):
        if debug:
            print(f'original Line: {txt_line}, original length: {len(txt_line)}' )         
        time_stamp = ''
        msg_source = ''
        blk_ids_regex = ''
        ip_address = ''
        signs_n_punctuations = ''
        white_space = ''

        if self.rm_time_stamp:
            time_stamp = '^\d+\s\d+\s\d+' 
        if self.rm_msg_source:
            msg_source = 'dfs\.\w+[$]\w+:|dfs\.\w+:'
        if self.rm_blk_ids_regex:
           # blk_ids_regex = 'blk_-\d+\.?'
           blk_ids_regex = 'blk_-?\d+\.?'
        if self.rm_ip_address:
            ip_address = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:*\d*'
        if self.rm_signs_n_punctuations:
            signs_n_punctuations = '\]|\[|\)|\(|\=|\,|\;|\/'
        if self.rm_white_space:
            white_space = '\s'
            
        pat = f'{time_stamp}|{msg_source}|{blk_ids_regex}|{ip_address}|{signs_n_punctuations}|{white_space}'     
        s = re.sub(pat, '', txt_line)
        if debug:
            print(f'cleaned line: {s},  cleaned length: {len(s)}')
            print()
        return s
        
    
    
    def train_char_tokenizer(self):
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        st_time = time.time()
        print('starting training the tokenizer:')
        tk.fit_on_texts(self.logs)
        end_time = time.time()
        print('ending tokenizer training:' , end_time - st_time)
        print('vocabulary size:' , len(tk.word_index))
        print('RAM usage: ', sys.getsizeof(tk) )
        self.tk = tk
        return tk
    
    def convert_char_to_numbers(self):
        print('starting text to number conversion')
        st_time = time.time()
        text_to_number = self.tk.texts_to_sequences(self.logs)
        end_time = time.time()
        print('ending text to number conversion:' , end_time - st_time)
        # padded_txt_to_num = pad_sequences(text_to_number, maxlen=256, padding='post')
        # when maxlen is not given, pad_sequences will calculate it automatically
        st_time = time.time()
        padded_txt_to_num = pad_sequences(text_to_number, maxlen=self.padded_char_len, 
                                          padding=self.padding_style, truncating=self.truncating)
        end_time = time.time()
        print('ending padding characters:' , end_time - st_time)
        print('padded_txt_to_num shape:', padded_txt_to_num.shape) # padded_txt_to_num shape: (11175629, 320)
        self.padded_txt_to_num = padded_txt_to_num
        print('RAM usage: ', sys.getsizeof(padded_txt_to_num), )
        return padded_txt_to_num
    
    def group_logs_nums_by_blkids(self):
        data_dict = OrderedDict()
        st_time = time.time()   
        for i, line in enumerate(self.logs):
           blkId_list = re.findall(r'(blk_-?\d+)', line)
           blkId_list = list(set(blkId_list))
           if len(blkId_list) >=2:
              continue
           blkId_set = set(blkId_list)
           for blk_Id in blkId_set:
             if not blk_Id in data_dict:
                 data_dict[blk_Id] = []
             data_dict[blk_Id].append(self.padded_txt_to_num[i])
             # if i % 100000: print(blk_Id, data_dict[blk_Id])
             if i % 1000000 == 0: 
                 print('completed: ', i)
                 end_time = time.time()
                 print('ending blk sequencing:' , end_time - st_time)     
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'LogSequence'])
        self.seq_of_log_nums = data_df
        print('RAM usage: ', sys.getsizeof(data_df) )
        return data_df
    
    def group_logs_texts_by_blkids(self):
        data_dict = OrderedDict()
        st_time = time.time()   
        for i, line in enumerate(self.logs):
           blkId_list = re.findall(r'(blk_-?\d+)', line)
           blkId_list = list(set(blkId_list))
           if len(blkId_list) >=2:
              continue
           blkId_set = set(blkId_list)
           for blk_Id in blkId_set:
             if not blk_Id in data_dict:
                 data_dict[blk_Id] = []
             data_dict[blk_Id].append(self.logs[i])
             # if i % 100000: print(blk_Id, data_dict[blk_Id])
             if i % 1000000 == 0: 
                 print('completed: ', i)
                 end_time = time.time()
                 print('ending blk sequencing:' , end_time - st_time)     
        df_seq_of_log_texts = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'LogSequence'])
        self.seq_of_log_texts = df_seq_of_log_texts
        print('RAM usage: ', sys.getsizeof(df_seq_of_log_texts) )
        return df_seq_of_log_texts
    
    def label_the_blk_txt_seq(self):
        label_data = pd.read_csv(self.labelfile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        label_data = '' #release memory
        self.seq_of_log_texts['Label'] = self.seq_of_log_texts['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        print('RAM usage: ', sys.getsizeof(self.seq_of_log_nums))
        return self.seq_of_log_nums
    
    def label_the_blk_num_seq(self):
        label_data = pd.read_csv(self.labelfile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        label_data = '' #release memory
        self.seq_of_log_nums['Label'] = self.seq_of_log_nums['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)  
        print('RAM usage: ', sys.getsizeof(self.seq_of_log_nums) )
        return self.seq_of_log_nums
    
        
    def train_test_split(self, x_data, y_data=None, ablation=0):
        # (x_data, y_data) = shuffle(x_data, y_data)
        if self.split_type == 'uniform' and y_data is not None:
            pos_idx = y_data > 0
            x_pos = x_data[pos_idx]
            y_pos = y_data[pos_idx]
            x_neg = x_data[~pos_idx]
            y_neg = y_data[~pos_idx]
            train_pos = int(self.train_ratio * x_pos.shape[0])
            train_neg = train_pos
            if ablation !=0:
                print(f'getting ablation data: {ablation}')
                train_pos = ablation
                train_neg = ablation
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
        # indexes = shuffle(np.arange(x_train.shape[0]))
        # x_train = x_train[indexes]
#         if y_train is not None:
#             y_train = y_train[indexes]
        print(y_train.sum(), y_test.sum()) # 8419 8419
        return x_train, y_train, x_test, y_test
        
    def make_equal_len_sequences(self, x_train, y_train, x_test, y_test):
        for i in range(5):
            print('length of train  sequence original', len(x_train[i]))            
        padded_x_train = pad_sequences(x_train, maxlen=self.padded_seq_len, 
                                       padding=self.padding_style, truncating=self.truncating)  # 57 taken automatically
        ##(16838, 57, 230)        
        for i in range(5):
            print('length of train sequence padded', len(padded_x_train[i]))            
        padded_x_test = pad_sequences(x_test, maxlen=self.padded_seq_len, 
                                      padding=self.padding_style, truncating=self.truncating) 
        # # # (558223, 57, 230)
        for i in range(5):
            print('len of test seq after padding',len(padded_x_test[i]))
        if self.save_train_test_data is True:
            with open('data\padded_train.pkl' , 'wb') as f:
                pickle.dump((padded_x_train, y_train), f)
            with open('data\padded_test.pkl' , 'wb') as f:
                pickle.dump((padded_x_test, y_test), f)
        return padded_x_train, y_train, padded_x_test, y_test
    
    
if __name__ == '__main__':
    hdfslog = HDFSLog()
    x_train, y_train, x_test, y_test = hdfslog.get_train_test_data()