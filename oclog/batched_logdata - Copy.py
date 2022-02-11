# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:51:26 2022

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

import tensorflow as tf

class HDFSLogGen:
    
    def __init__(self, logpath='C:\ML_data\Logs', 
                 labelpath='C:\ML_data\Logs',
                 logfilename='HDFS.log',
                 labelfilename='anomaly_label.csv',):
        
        self.logpath = logpath
        self.labelpath = labelpath
        self.logfilename = logfilename
        self.labelfilename = labelfilename
        self.logfile = os.path.join(logpath, logfilename)
        self.labelfile = os.path.join(labelpath, labelfilename)    
    
    
    def get_log_lines(self):
        with open(self.logfile, 'r', encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip().lower() for x in logs]
        n_logs = len(logs)
        print('total number of lines in the log file:', n_logs)
        return logs
    
    def get_label_dict(self):
        label_data = pd.read_csv(self.labelfile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        return label_dict
    
    def get_sequence_byid_with_label(self, save_pkl=False):
        loglines = self.get_log_lines()
        label_dict = self.get_label_dict()
        blkid_dict = OrderedDict()
        st_time = time.time()   
        for i, line in enumerate(loglines):
           blkId_list = re.findall(r'(blk_-?\d+)', line)
           blkId_list = list(set(blkId_list))
           if len(blkId_list) >=2:
              continue
           blkId_set = set(blkId_list)
           for blk_Id in blkId_set:
             if not blk_Id in blkid_dict:
                 blkid_dict[blk_Id] = ([], 1 if label_dict.get(blk_Id)=='Anomaly' else 0)
             blkid_dict[blk_Id][0].append(line)
             # if i % 100000: print(blk_Id, data_dict[blk_Id])
             if i % 1000000 == 0: 
                 print('completed: ', i)
                 end_time = time.time()
                 print('ending blk sequencing:' , end_time - st_time)
        if save_pkl is True:
            with open('data\hdfs_sequence_byid_with_label.pkl' , 'wb') as f:
                pickle.dump(blkid_dict, f)
        return blkid_dict
    
    def group_by_blkid(self, loglines, label_dict):
        blkid_dict = OrderedDict()
        st_time = time.time()   
        for i, line in enumerate(loglines):
           blkId_list = re.findall(r'(blk_-?\d+)', line)
           blkId_list = list(set(blkId_list))
           if len(blkId_list) >=2:
              continue
           blkId_set = set(blkId_list)
           for blk_Id in blkId_set:
             if not blk_Id in blkid_dict:
                 blkid_dict[blk_Id] = []
             blkid_dict[blk_Id].append(line)
             # if i % 100000: print(blk_Id, data_dict[blk_Id])
             if i % 1000000 == 0: 
                 print('completed: ', i)
                 end_time = time.time()
                 print('ending blk sequencing:' , end_time - st_time)             
        return blkid_dict

if __name__ == '__main__':
    hdfsdata = HDFSLogGen()    
    logs = hdfsdata.get_sequence_byid_with_label()
    seq = [v[0] for v in logs.values()]
    label = [v[1] for v in logs.values()]
    tflogs = tf.data.Dataset.from_tensor_slices((seq, label))
    
        

        
    