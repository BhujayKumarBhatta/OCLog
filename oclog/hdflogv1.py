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


class HDFSLogv1:
    
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
        self.rm_time_stamp=rm_time_stamp
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
        self.cleaned_logs = None
        self.blkid_to_line_to_num = None
                
        
    def get_log_lines(self):
        st_time = time.time()
        with open(self.logfile, 'r', encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip().lower() for x in logs]
        n_logs = len(logs)
        print('total number of lines in the log file:', n_logs)
        print('RAM usage: ', sys.getsizeof(logs) )
        self.logs = logs
        end_time = time.time()
        print('ending logs in memory:' , end_time - st_time)
        return logs  
    
    def remove_unwanted_characters_n_words(self, txt_line, debug=False):
        # if debug:
        #     print(f'original Line: {txt_line}, original length: {len(txt_line)}' )         
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
        # print(type(s))
        # if debug:
        #     print(f'cleaned line: {s},  cleaned length: {len(s)}')
        #     print()
        return s
    
    def get_blkid_n_clean_text(self):
        if self.logs is None:
            self.get_log_lines()
        st_time = time.time()
        cleaned_logs = []
        for i, line in enumerate(self.logs):
            blkId_list = re.findall(r'(blk_-?\d+)', line)
            blkId_list = list(set(blkId_list))
            if len(blkId_list) >=2:
                continue
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                tup = (blk_Id, self.remove_unwanted_characters_n_words(line))
                cleaned_logs.append(tup)
        self.cleaned_logs = cleaned_logs
        end_time = time.time()
        print('loaded cleaned logs with blk_id  in memory:' , end_time - st_time)
        print('RAM usage: ', sys.getsizeof(cleaned_logs) )
        return cleaned_logs
    
    def get_cleaned_txt_without_blkid(self):
        if self.cleaned_logs is None:
            self.get_blkid_n_clean_text()
        st_time = time.time()
        cleaned_logs_witout_blkid =  [tup[1] for tup in self.cleaned_logs]
        end_time = time.time()
        print('loaded cleaned logs without blkid in memory:' , end_time - st_time)
        print('RAM usage: ', sys.getsizeof(self.cleaned_logs) )
        return cleaned_logs_witout_blkid
        
    
    def train_char_tokenizer(self):
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        st_time = time.time()
        cleaned_logs_witout_blkid = self.get_cleaned_txt_without_blkid()
        print('starting training the tokenizer:')
        tk.fit_on_texts(cleaned_logs_witout_blkid)
        end_time = time.time()
        print('ending tokenizer training:' , end_time - st_time)
        print('RAM usage: ', sys.getsizeof(tk) )
        print('vocabulary size:' , len(tk.word_index))
        self.tk = tk
        return tk
    
    def convert_char_to_numbers(self):
        if tk is None:
            self.train_char_tokenizer()
        if self.cleaned_logs is None:
            self.get_blkid_n_clean_text()
        print('starting text to number conversion')
        st_time = time.time()
        blkid_to_line_to_num = []
        for i, (blkid, line) in enumerate(self.cleaned_logs):
            # don't put line without [] - txt_2_num = self.tk.texts_to_sequences(line])
            # this will convert each character to sequence 
            txt_2_num = self.tk.texts_to_sequences([line])
            padded_txt_to_num = pad_sequences(txt_2_num, maxlen=self.padded_char_len, 
                                              padding=self.padding_style, truncating=self.truncating)
            blkid_to_line_to_num.append((blkid, padded_txt_to_num[0])) 
            if i % 1000000 == 0: 
                print('completed: ', i)
                end_time = time.time()
                print('time :' , end_time - st_time) 
        end_time = time.time()
        print('ending text to number conversion:' , end_time - st_time)        
        print('RAM usage: ', sys.getsizeof(blkid_to_line_to_num), )
        self.blkid_to_line_to_num = blkid_to_line_to_num
        return blkid_to_line_to_num
    
    
                
if __name__ == '__main__':
    hdfslogs = HDFSLogv1(padded_seq_len=32,
                         padded_char_len=64,)
    clogs = hdfslogs.get_blkid_n_clean_text()
    tk = hdfslogs.train_char_tokenizer()
    blkid_to_line_to_num = hdfslogs.convert_char_to_numbers()
    print(blkid_to_line_to_num[0][1])
    
    
########################## correct result
# print(blkid_to_line_to_num[0][1])
# [ 4  7 13  3 10  2 11  2  4 23  4  7 24 15 12  3 11 14  8 10 11 19  5  2
#   8  6 19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

###### wrong result ###############

                 