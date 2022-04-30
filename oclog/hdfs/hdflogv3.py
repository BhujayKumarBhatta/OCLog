import os
import sys
import re
import psutil
import time
import pickle
import random
from collections import OrderedDict
from itertools  import groupby
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

random.seed(123)
tf.random.set_seed(123)


class HDFSLogv3:
    
    def __init__(self, **kwargs ):
        
        #### either parameters or methods will assign them ###########
        self.hdfs_logpath = kwargs.get('hdfs_logpath', 'C:\ML_data\Logs')
        self.hdfs_logfilename = kwargs.get('hdfs_logfilename', 'HDFS.log')       
        self.hdfs_labelpath = kwargs.get('hdfs_labelpath', 'C:\ML_data\Logs')
        self.hdfs_labelfilename = kwargs.get('hdfs_labelfilename', 'anomaly_label.csv') 
        
        self.hdfs_rm_time_stamp = kwargs.get('hdfs_rm_time_stamp', True)        
        self.hdfs_rm_blk_ids_regex = kwargs.get('hdfs_rm_blk_ids_regex', True) 
        self.hdfs_rm_ip_address = kwargs.get('hdfs_rm_ip_address', True) 
        self.hdfs_rm_msg_source = kwargs.get('hdfs_rm_msg_source', True) 
        self.padded_char_len = kwargs.get('padded_char_len', 64)
        self.padding_style_char = kwargs.get('padding_style_char', 'post')
        self.truncating_style_char = kwargs.get('truncating_style_char', 'pre') 
        self.padded_seq_len = kwargs.get('padded_seq_len', 32)
        self.padding_style_seq = kwargs.get('padding_style_seq', 'post')
        self.truncating_style_seq = kwargs.get('truncating_style_seq', 'pre')
        # self.hdfs_labelpath = kwargs.get('hdfs_labelpath', 'C:\ML_data\Logs')
        # self.hdfs_labelfilename = kwargs.get('hdfs_labelfilename', 'anomaly_label.csv') 
        self.hdfs_rm_signs_n_punctuations = kwargs.get('hdfs_rm_signs_n_punctuations', True)
        self.hdfs_rm_white_space = kwargs.get('hdfs_rm_white_space', True)        
        self.hdfs_obj_save_path = kwargs.get('hdfs_obj_save_path', 'data')
        self.debug = kwargs.get('debug', False)
        
        #### methods will assign them #####
        self.tk = None    
        self.lebeled_num_seq_df_epn = None
        self.hdfs_saved_obj_name = None
        
        
        
    
    ### 1st - done
    def get_log_lines(self, **kwargs):
        '''
        hdfs_logpath =  'C:\ML_data\Logs'
        hdfs_logfilename = 'HDFS.log'        
        '''
        st_time = time.time()
        hdfs_logpath = kwargs.get('hdfs_logpath', self.hdfs_logpath)
        hdfs_logfilename = kwargs.get('hdfs_logfilename', self.hdfs_logfilename)
        hdfs_logfile = os.path.join(hdfs_logpath, hdfs_logfilename)
        with open(hdfs_logfile, 'r', encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip().lower() for x in logs]
        n_logs = len(logs)
        if self.debug:
            print('total number of lines in the log file:', n_logs)
            print('RAM usage: ', sys.getsizeof(logs) )
        # self.logs = logs
        end_time = time.time()
        if self.debug:
            print('loaded logs in memory in time:' , end_time - st_time)
        return logs  
    
    
    ### 2nd - No storage
    def remove_unwanted_characters_n_words(self, txt_line, **kwargs):
        '''
        hdfs_rm_time_stamp = True
        hdfs_rm_msg_source =  True
        hdfs_rm_blk_ids_regex = True
        hdfs_rm_ip_address = True
        hdfs_rm_signs_n_punctuations = True
        hdfs_rm_white_space = True
        '''
        # if debug:
        #     print(f'original Line: {txt_line}, original length: {len(txt_line)}' ) 
        hdfs_rm_time_stamp = kwargs.get('hdfs_rm_time_stamp', self.hdfs_rm_time_stamp)
        self.hdfs_rm_time_stamp = hdfs_rm_time_stamp ### will be used later for creating name of the pickle file
        
        hdfs_rm_msg_source = kwargs.get('hdfs_rm_msg_source', self.hdfs_rm_msg_source)
        hdfs_rm_blk_ids_regex = kwargs.get('hdfs_rm_blk_ids_regex', self.hdfs_rm_blk_ids_regex)
        hdfs_rm_ip_address = kwargs.get('hdfs_rm_ip_address', self.hdfs_rm_ip_address)
        self.hdfs_rm_ip_address = hdfs_rm_ip_address  ### will be used later for creating name of the pickle file
        
        hdfs_rm_signs_n_punctuations = kwargs.get('hdfs_rm_signs_n_punctuations', self.hdfs_rm_signs_n_punctuations)
        hdfs_rm_white_space = kwargs.get('hdfs_rm_white_space', self.hdfs_rm_white_space)        
        time_stamp = ''
        msg_source = ''
        blk_ids_regex = ''
        ip_address = ''
        signs_n_punctuations = ''
        white_space = ''

        if hdfs_rm_time_stamp:
            time_stamp = '^\d+\s\d+\s\d+' 
        if hdfs_rm_msg_source:
            msg_source = 'dfs\.\w+[$]\w+:|dfs\.\w+:'
        if hdfs_rm_blk_ids_regex:
           # blk_ids_regex = 'blk_-\d+\.?'
           blk_ids_regex = 'blk_-?\d+\.?'
        if hdfs_rm_ip_address:
            ip_address = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:*\d*'
        if hdfs_rm_signs_n_punctuations:
            signs_n_punctuations = '\]|\[|\)|\(|\=|\,|\;|\/'
        if hdfs_rm_white_space:
            white_space = '\s'
            
        pat = f'{time_stamp}|{msg_source}|{blk_ids_regex}|{ip_address}|{signs_n_punctuations}|{white_space}'     
        s = re.sub(pat, '', txt_line)
        # print(type(s))
        # if debug:
        #     print(f'cleaned line: {s},  cleaned length: {len(s)}')
        #     print()
        return s
    
    
    ### 3rd done 
    def get_blkid_n_clean_text(self, logs, **kwargs):        
        # logs = self.get_log_lines()
        st_time = time.time()
        cleaned_logs = []
        for i, line in enumerate(logs):
            blkId_list = re.findall(r'(blk_-?\d+)', line)
            blkId_list = list(set(blkId_list))
            if len(blkId_list) >=2:
                continue
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                tup = (blk_Id, self.remove_unwanted_characters_n_words(line, **kwargs))
                cleaned_logs.append(tup)
        # self.cleaned_logs = cleaned_logs
        end_time = time.time()
        if self.debug:
            print('loaded cleaned logs with blk_id  in memory:' , end_time - st_time)
            print('RAM usage: ', sys.getsizeof(cleaned_logs) )
        return cleaned_logs
    
    
    
    ### 4th - no storage
    def get_cleaned_txt_without_blkid(self, cleaned_logs, **kwargs):
        # cleaned_logs = self.get_blkid_n_clean_text()
        st_time = time.time()
        cleaned_logs_witout_blkid =  [tup[1] for tup in cleaned_logs]
        end_time = time.time()
        if self.debug:
            print('loaded cleaned logs without blkid in memory:' , end_time - st_time)
            print('RAM usage: ', sys.getsizeof(cleaned_logs) )
        return cleaned_logs_witout_blkid
    
    
    ### 5th  - minor storage
    def train_char_tokenizer(self, cleaned_logs_witout_blkid, **kwargs):
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        st_time = time.time()
        # cleaned_logs_witout_blkid = self.get_cleaned_txt_without_blkid()
        if self.debug:
            print('starting training the tokenizer:')
        tk.fit_on_texts(cleaned_logs_witout_blkid)
        end_time = time.time()
        if self.debug:
            print('ending tokenizer training:' , end_time - st_time)
            print('RAM usage: ', sys.getsizeof(tk) )
            print('vocabulary size:' , len(tk.word_index))
        self.tk = tk
        return tk
    
    
    
    
    ### 6th - done
    def convert_char_to_numbers(self, cleaned_logs, **kwargs):
        '''
        Identifies all the unique characters in the corpus and builds a character to number map
        Then all chars in a lines is converted to number as per the map
        To make all the line equal lenght a padding and trucating strategy is applied
        The padded_char_len should be same as bglog if you want to mis the hdfs and bglog.
        other parameters:
        padding_style_char = kwargs.get('padding_style_char','post')
        truncating_style_char = kwargs.get('truncating_style_char', 'pre')    
        '''
       
        padded_char_len = kwargs.get('padded_char_len', self.padded_char_len)
        self.padded_char_len = padded_char_len ### since this will be used in the name of pickle file 
        padding_style_char = kwargs.get('padding_style_char', self.padding_style_char)
        truncating_style_char = kwargs.get('truncating_style_char', self.truncating_style_char)        
                    
        if self.tk is None:
            tk =  self.train_char_tokenizer()
        # cleaned_logs = self.get_blkid_n_clean_text()
        if self.debug:
            print('starting text to number conversion')
        st_time = time.time()
        blkid_to_line_to_num = []
        for i, (blkid, line) in enumerate(cleaned_logs):
            # don't put line without [] - txt_2_num = self.tk.texts_to_sequences(line])
            # this will convert each character to sequence 
            txt_2_num = self.tk.texts_to_sequences([line])
            padded_txt_to_num = pad_sequences(txt_2_num, maxlen=padded_char_len, 
                                              padding=padding_style_char, truncating=truncating_style_char)
            blkid_to_line_to_num.append((blkid, padded_txt_to_num[0])) 
            if i % 1000000 == 0 and self.debug: 
                print('completed: ', i)
                end_time = time.time()
                print('time :' , end_time - st_time) 
        end_time = time.time()
        if self.debug:
            print('ending text to number conversion:' , end_time - st_time)        
            print('RAM usage: ', sys.getsizeof(blkid_to_line_to_num), )
        # self.blkid_to_line_to_num = blkid_to_line_to_num
        return blkid_to_line_to_num
    
        
    
    ### 8th - done 
    def get_num_seq_by_blkid_itertools(self, blkid_to_line_to_num, **kwargs):
        st_time = time.time()
        # if self.blkid_to_line_to_num is None:
        # blkid_to_line_to_num =  self.convert_char_to_numbers()
        result = { k : [*map(lambda v: v[1], values)]
            for k, values in groupby(sorted(blkid_to_line_to_num, key=lambda x: x[0]), lambda x: x[0])
            }        
        end_time = time.time()
        if self.debug:
            print('ending num_sequence_by_blkid conversion:' , end_time - st_time)        
            print('RAM usage: ', sys.getsizeof(result) )
        num_seq_by_blkid_itertools = result
        return num_seq_by_blkid_itertools
    
    
    ### 9th  this is what is required 
    def get_labelled_num_seq_df(self, num_seq_by_blkid_itertools, **kwargs):
        '''
        the sequences are converted to datframe
        the sequences are made fixed length, meaning fixed number of lines in all sequences by applying a padding and truncating strategy
        so the padding and truncating is actually applied twice, once duting the char to num conversion in previous methods and 
        another duting this mehod. 
        Choose the parameter, padded_seq_len same as bgl and hdfs if you want to mix the logs:
        other params:
        padding_style_seq = default is 'post'
        truncating_style_seq = default is 'pre'
        hdfs_labelpath = default is  'C:\ML_data\Logs'
        hdfs_labelfilename = default is 'anomaly_label        
        
        '''
        
        st_time = time.time()        
        padded_seq_len = kwargs.get('padded_seq_len', self.padded_seq_len)
        self.padded_seq_len = padded_seq_len ### since this will be used while saving the pickle file
        padding_style_seq = kwargs.get('padding_style_seq', self.padding_style_seq)
        self.padding_style_seq = padding_style_seq
        truncating_style_seq = kwargs.get('truncating_style_seq', self.truncating_style_seq)
        hdfs_labelpath = kwargs.get('hdfs_labelpath', self.hdfs_labelpath)
        hdfs_labelfilename = kwargs.get('hdfs_labelfilename', self.hdfs_labelfilename)        
        hdfs_lalbelfile = os.path.join(hdfs_labelpath, hdfs_labelfilename)
        # num_seq_by_blkid_itertools = self.get_num_seq_by_blkid_itertools()
        ######### changes: chaging the column name  as per bgl log  ##################
        labelled_num_seq_df = pd.DataFrame(list(num_seq_by_blkid_itertools.items()), columns=['BlockId', 'seq'])
        label_data = pd.read_csv(hdfs_lalbelfile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict() # The file heading is 'Label' , capital L
        ######## changes: Don't want to convert the label to 0 or 1 , 
        ##### we want the text label and distingused as hdfs_anomaly and hdfs_normal####
        labelled_num_seq_df['label'] = labelled_num_seq_df['BlockId'].apply(
            lambda x: "hdfs_anomaly" if label_dict[x] == 'Anomaly' else "hdfs_normal")
        ##### changes: pad the sequence here itself to make all the sequence of equal length ############
        ### get the seq column and run pad_sequence
        padded_seq = pad_sequences(labelled_num_seq_df.seq, maxlen=padded_seq_len, 
                                   padding=padding_style_seq, truncating=truncating_style_seq)
        ### the result will be a three dimensiona np array (rows, lines in a seq(32), chars in line(176))
        ### convert that to a list since pandas series cant take 3D data and then convert to pandas series
        padded_seq_series = pd.Series(list(padded_seq))
        ### Drop the original col and unnecessary bolid col
        labelled_num_seq_df.drop(columns=['seq', 'BlockId'], inplace=True)
        #### order the cols so that seq appears before label similar to bglog
        labelled_num_seq_df['seq'] = padded_seq_series
        labelled_num_seq_df = labelled_num_seq_df[['seq', 'label']]
        end_time = time.time()
        if self.debug:
            print('ending labelled_num_seq_df conversion:' , end_time - st_time) 
            print('RAM usage: ', sys.getsizeof(labelled_num_seq_df) )
        return labelled_num_seq_df
    
                
    def get_lebeled_num_seq_df_epn(self, **kwargs):
        '''
        epn menas equal positive and negative class
        HDFS log has a large number of negative class, 558223 as against negative class, 16830
        To make the processed data smaller we are taking 16830 each from positive and negative class
        The output df has similar structure as bglog, hence this can be later used to concatenate and 
        create a mix data where bgl classes can be extended from 6 classes to 8 classes. 
        '''
        logs = self.get_log_lines(**kwargs)
        cleaned_logs = self.get_blkid_n_clean_text(logs, **kwargs)
        cleaned_logs_witout_blkid = self.get_cleaned_txt_without_blkid(cleaned_logs, **kwargs)
        tk = self.train_char_tokenizer(cleaned_logs_witout_blkid, **kwargs)
        blkid_to_line_to_num = self.convert_char_to_numbers(cleaned_logs, **kwargs)
        # num_sequence_by_blkid = self.get_num_sequence_by_blkid(blkid_to_line_to_num, **kwargs)
        num_seq_by_blkid_itertools = self.get_num_seq_by_blkid_itertools(blkid_to_line_to_num, **kwargs)
        labelled_num_seq_df = self.get_labelled_num_seq_df(num_seq_by_blkid_itertools, **kwargs)   
        
        # hdfs_obj_load_fm_disk = kwargs.get('hdfs_obj_load_fm_disk', False)
        # hdfs_saved_obj_name = kwargs.get('hdfs_saved_obj_name', self.hdfs_saved_obj_name )
        # if hdfs_obj_load_fm_disk:
        #     with open(hdfs_saved_obj_name, 'rb') as f:
        #         eq_neg_pos_df = pickle.load(f)
        # else:
        # labelled_num_seq_df = self.get_labelled_num_seq_df()
        # if self.lebeled_num_seq_df_epn is not None:
        #     lebeled_num_seq_df_epn = self.lebeled_num_seq_df_epn
        # else:
        #     labelled_num_seq_df = self.get_labelled_num_seq_df()
        pos_df = labelled_num_seq_df[labelled_num_seq_df.label == 'hdfs_anomaly']
        if self.debug: print('len of pos_df', len(pos_df))
        neg_df = labelled_num_seq_df[labelled_num_seq_df.label == 'hdfs_normal']
        if self.debug: print('len of neg_df', len(neg_df))
        neg_df_eq_pos_df = neg_df[:len(pos_df)]        
        lebeled_num_seq_df_epn = pd.concat([pos_df, neg_df_eq_pos_df])        
        print('eq_neg_pos_df shape:',lebeled_num_seq_df_epn.shape)
        self.lebeled_num_seq_df_epn = lebeled_num_seq_df_epn
        return lebeled_num_seq_df_epn     
    
    
    
    def get_train_test_split_single_class(self, label=0, **kwargs):
        '''
        ablation = 1000
        train_ratio = 0.8
        val_ratio = None
        test_ratio = None
        '''
        ablation = kwargs.get('ablation', 1000)
        train_ratio = kwargs.get('train_ratio', 0.8)
        val_ratio = kwargs.get('val_ratio', None)
        test_ratio = kwargs.get('test_ratio', None)
        hdfsdf = self.lebeled_num_seq_df_epn
        if hdfsdf is None: 
            hdfsdf = self.get_lebeled_num_seq_df_epn(**kwargs)
        train_data = None
        val_data = None
        test_data = None
        train_cnt = round(ablation * train_ratio)#### 1000 * 0.7 = 700
        remaining_cnt = round(ablation *(1 - train_ratio)) ### 1000 * (1-0.7) = 300 
        if val_ratio is None and test_ratio is None:
            val_cnt = test_cnt = remaining_cnt//2 ### 300/2 = 150 each
        else:
            val_cnt = round(ablation * val_ratio) ### 1000 * 0.2 = 200 
            test_cnt = round(ablation * test_ratio) ### 1000 * 0.1 = 100
        
        cls_data = hdfsdf[hdfsdf.label==label]
        cls_data_cnt = cls_data.count()[0]
        # cls_unique_label = np.unique(cls_data.label)[0]
        #if self.debug: print('cls_unique_label', cls_unique_label)
        if ablation <= cls_data_cnt: ### if 1000 <= 2000            
                train_data = cls_data[0:train_cnt] ### cls_data[0:700]
                val_data = cls_data[train_cnt:train_cnt+val_cnt] ### cls_data[700:(700+200)]
                test_data = cls_data[train_cnt+val_cnt:ablation] ### cls_data[900:1000]
        elif ablation > cls_data_cnt and cls_data_cnt >= train_cnt+val_cnt: ### 1000>950 and 950>(700+200)
            train_data = cls_data[0:train_cnt] ### cls_data[0:700]
            remaining_for_test = cls_data_cnt - (train_cnt+val_cnt) ### 950 - (700+200) = 50
            if remaining_for_test > 0: ### 50 > 0
                val_data = cls_data[train_cnt:train_cnt+val_cnt] ### cls_data[700:(700+200)]
                test_data = cls_data[train_cnt+val_cnt:cls_data_cnt] ### cls_data[900:950]
            else: ### cls_data_cnt = 850 or 900
                val_data = cls_data[train_cnt:cls_data_cnt] ### cls_data[700:850]
        else:
            if self.debug:
                    print(f'{cls_data_cnt} data in class {label} not enough to split into train:{train_cnt}'
                          f'and validation:{val_cnt}, adding the entire data as ukc')
        if train_data is not None:
            print(f'train_{label}:, {train_data.count()[0]}', end=', ')
        if val_data is not None:
            print(f'val_{label}:, {val_data.count()[0]}', end=', ')
        if test_data is not None:
            print(f'test_{label}:, {test_data.count()[0]}', end=', ')       
        return train_data, val_data,test_data
    
            
    def get_train_test_multi_class(self, **kwargs):       
        classes=['hdfs_anomaly', 'hdfs_normal']        
        hdfsdf = self.lebeled_num_seq_df_epn
        if hdfsdf is None: 
            hdfsdf = self.get_lebeled_num_seq_df_epn(**kwargs)
            
        train_data, val_data, test_data = [], [], []
        for class_name in classes:
                trdata, valdata, tsdata = self.get_train_test_split_single_class(label=class_name, **kwargs)
                if trdata is not None: train_data.append(trdata)
                if valdata is not None: val_data.append(valdata)
                if tsdata is not None: test_data.append(tsdata)
        train_df = pd.concat(train_data)
        val_df = pd.concat(val_data)
        test_df = pd.concat(test_data)
        if self.debug: 
            print('train:', train_df.label.value_counts())
            print('val:', val_df.label.value_counts())
            print('test:', test_df.label.value_counts())
        return train_df, val_df, test_df 
        
    
    def get_train_test_binary(self, **kwargs):
        '''
        converts the labels as : hdfs_normal = 0 and hdfs_anomaly = 1
        return a tule of data sets each one of them is a list
        train_test_binary = x_train, y_train, x_val, y_val, x_test, y_test
        '''
        train_df, val_df, test_df  = self.get_train_test_multi_class(**kwargs)
        binary_train_df = train_df.copy()
        binary_val_df = val_df.copy()
        binary_test_df = test_df.copy()
        binary_train_df['label'].replace({'hdfs_normal': 0, 'hdfs_anomaly': 1}, inplace=True)
        binary_val_df['label'].replace({'hdfs_normal': 0, 'hdfs_anomaly': 1}, inplace=True)
        binary_test_df['label'].replace({'hdfs_normal': 0, 'hdfs_anomaly': 1}, inplace=True)
        x_train = list(binary_train_df.seq.values)
        y_train = list(binary_train_df.label.values)         
        x_val = list(binary_val_df.seq.values)
        y_val = list(binary_val_df.label.values)        
        # print(y_val[:2])
        x_test = list(binary_test_df.seq.values)     
        y_test = list(binary_test_df.label.values)
        # unique_label_train = np.unique(binary_train_df.label)
        # max_label_num_train = max(unique_label_train)
        # ukc_label = str(int(max_label_num_train)+1)
        # self.test_df.loc[self.test_df.label > max_label_num_train, 'label' ]=ukc_label
        # y_test = to_categorical(y_test)        
        train_test_binary = x_train, y_train, x_val, y_val, x_test, y_test
        return train_test_binary
    
       
    
    def get_tensor_train_val_test(self, **kwargs):
        '''
        params: batch_size = 32
        returns tensorflow dataset
        '''
        ablation = kwargs.get('ablation', 1000)        
        B = kwargs.get('batch_size', 32)
        
        train_test_binary = self.get_train_test_binary(**kwargs)
        
        
        x_train, y_train, x_val, y_val, x_test, y_test = train_test_binary
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.shuffle(buffer_size=len(y_train)).batch(B, drop_remainder=True)
        
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.shuffle(buffer_size=len(y_val)).batch(B, drop_remainder=True)
        
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_data = test_data.shuffle(buffer_size=len(y_test)).batch(B, drop_remainder=True)
        
        if self.debug: 
            # if self.debug: print(self.padded_num_seq_df.label.value_counts())
            print('train_data',train_data)
            print('val_data', val_data)
            print('test_data', test_data)
            print('char in lines, train_data.element_spec[0].shape[2]',train_data.element_spec[0].shape[2])
            print('num classes, train_data.element_spec[1].shape: ',train_data.element_spec[1].shape)            
            print('length of val_data:',len(val_data))
        print('length of train_data - (num_seq_per_cls * num_class)// batch size:', len(train_data))
        tensor_train_val_test = train_data, val_data, test_data
        return tensor_train_val_test
    
    
    def save_hdfs_log_obj(self, **kwargs):
        full_file_name = None
        hdfs_obj_save = kwargs.get('hdfs_obj_save', True)
        hdfs_rm_time_stamp = kwargs.get('hdfs_rm_time_stamp', self.hdfs_rm_time_stamp)
        self.hdfs_rm_time_stamp = hdfs_rm_time_stamp ### for easy access to the calss attribute in the saved object
        hdfs_rm_ip_address = kwargs.get('hdfs_rm_ip_address', self.hdfs_rm_ip_address)
        self.hdfs_rm_ip_address = hdfs_rm_ip_address ### for easy access to the calss attribute in the saved object
        hdfs_obj_save_path = kwargs.get('hdfs_obj_save_path', self.hdfs_obj_save_path)
        self.hdfs_obj_save_path = hdfs_obj_save_path ### for easy access to the calss attribute in the saved object
        if hdfs_obj_save:
            hdfsdf = self.lebeled_num_seq_df_epn
            if hdfsdf is None: 
                hdfsdf = self.lebeled_num_seq_df_epn(**kwargs)
                
            if not self.hdfs_rm_time_stamp and not self.hdfs_rm_ip_address:
                meta_status = 'time_ip'
            elif not self.hdfs_rm_time_stamp and self.hdfs_rm_ip_address:
                meta_status = 'tstamp'
            elif self.hdfs_rm_time_stamp and not self.hdfs_rm_ip_address:
                meta_status = 'ip'
            else:
                meta_status = 'no_meta'            
            default_file_name = 'hdfsobj_' + str(self.padded_seq_len) + '_' + str(self.padded_char_len) +'_' + meta_status + '.pkl' 
            hdfs_obj_name = kwargs.get('hdfs_obj_name', default_file_name)

            if not os.path.exists(hdfs_obj_save_path):
                os.mkdir(hdfs_obj_save_path)
            full_file_name = os.path.join(hdfs_obj_save_path, hdfs_obj_name)
            full_file_name = os.path.abspath(full_file_name)
            self.hdfs_saved_obj_name = full_file_name
            with open(full_file_name, 'wb') as f:
                pickle.dump(self, f)
            print(f'saved hdfs object as {full_file_name}')
        return full_file_name
        
        
       
        
        
        

            

if __name__ == '__main__':
    hdfslogs = HDFSLogv1(padded_seq_len=32,
                         padded_char_len=64,)
    clogs = hdfslogs.get_blkid_n_clean_text()
    tk = hdfslogs.train_char_tokenizer()
    blkid_to_line_to_num = hdfslogs.convert_char_to_numbers()    
    print(blkid_to_line_to_num[0][1])
    # num_sequence_by_blkid = hdfslogs.get_num_sequence_by_blkid()
    # print(num_sequence_by_blkid['blk_4258862871822415442'])
    # print(len(num_sequence_by_blkid['blk_4258862871822415442']))
    num_seq_by_blkid_itertools = hdfslogs.get_num_seq_by_blkid_itertools()
    # print(num_seq_by_blkid_itertools['blk_4258862871822415442'])
    labelled_num_seq_df = hdfslogs.get_labelled_num_seq_df()
    # tk.sequences_to_texts(labelled_num_seq_df['LogNumSequence'][0])
    padded_x_train, y_train, padded_x_test, y_test = hdfslogs.get_padded_train_test_data()
    padded_train_test_data = padded_x_train, y_train, padded_x_test, y_test
    with open('data\padded_train_test_data.pkl', 'wb') as f:
        pickle.dump(padded_train_test_data, f)
    train_test_data = hdfslogs.train_test_data
    with open('data\train_test_data.pkl', 'wb') as f:
        pickle.dump(train_test_data, f)
        

#################################################################################################
########################## correct result
# print(blkid_to_line_to_num[0][1])
# [ 4  7 13  3 10  2 11  2  4 23  4  7 24 15 12  3 11 14  8 10 11 19  5  2
#   8  6 19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

###### wrong result ###############

########### a log line mapped with ID now looks like 
# ('blk_-1608999687919862906', array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  4,  7, 24, 15, 12,  3, 11,
#        14,  8, 10, 11, 19,  5,  2,  8,  6, 19,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))

##### how the log text is mapped with number ############
# clogs[155:158]
# Out[14]: 
# [('blk_-3544583377289625738', 'infoservedblockto'),
#  ('blk_-1608999687919862906',
#   'infoblock*namesystem.addstoredblock:blockmapupdated:isaddedtosize91178'),
#  ('blk_-1608999687919862906',
#   'infoblock*namesystem.addstoredblock:blockmapupdated:isaddedtosize91178')]

# print(blkid_to_line_to_num[155:158])
# [('blk_-3544583377289625738', array([ 4,  7, 13,  3,  8,  2, 10, 23,  2,  5, 15, 12,  3, 11, 14,  6,  3,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])), ('blk_-1608999687919862906', array([ 3, 11, 14, 31,  7,  9, 16,  2,  8, 28,  8,  6,  2, 16, 27,  9,  5,
#         5,  8,  6,  3, 10,  2,  5, 15, 12,  3, 11, 14, 19, 15, 12,  3, 11,
#        14, 16,  9, 18, 25, 18,  5,  9,  6,  2,  5, 19,  4,  8,  9,  5,  5,
#         2,  5,  6,  3,  8,  4, 32,  2, 37, 21, 21, 30, 20])), ('blk_-1608999687919862906', array([ 3, 11, 14, 31,  7,  9, 16,  2,  8, 28,  8,  6,  2, 16, 27,  9,  5,
#         5,  8,  6,  3, 10,  2,  5, 15, 12,  3, 11, 14, 19, 15, 12,  3, 11,
#        14, 16,  9, 18, 25, 18,  5,  9,  6,  2,  5, 19,  4,  8,  9,  5,  5,
#         2,  5,  6,  3,  8,  4, 32,  2, 37, 21, 21, 30, 20]))]


#############sequence to text checking ###############################
# labelled_num_seq_df['LogNumSequence'][0]
# Out[4]: 
# [array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  4,  7, 24, 15, 12,  3, 11,
#         14,  8, 10, 11, 19,  5,  2,  8,  6, 19,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  4,  7, 24, 15, 12,  3, 11,
#         14,  8, 10, 11, 19,  5,  2,  8,  6, 19,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  4,  7, 24, 15, 12,  3, 11,
#         14,  8, 10, 11, 19,  5,  2,  8,  6, 19,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([10,  3,  3,  6, 10,  9,  7,  5, 20, 29,  6,  2, 16, 18,  3, 10,  9,
#         10, 28, 29,  6,  9,  8, 14, 29, 33, 17, 17, 20, 21, 21, 21, 17, 21,
#         17, 33, 26, 29, 17, 17, 21, 36, 29, 16, 29, 17, 17, 21, 33, 22, 21,
#         29, 17, 18,  9, 10,  6, 40, 17, 21, 33, 22, 21, 27]),
#  array([ 4,  7, 13,  3, 18,  9, 11, 14,  2,  6, 10,  2,  8, 18,  3,  7,  5,
#          2, 10, 33, 13,  3, 10, 15, 12,  3, 11, 14,  6,  2, 10, 16,  4,  7,
#          9,  6,  4,  7, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  2,  5, 15, 12,  3, 11, 14,
#          3, 13,  8,  4, 32,  2, 35, 36, 36, 35, 33, 26, 21, 13, 10,  3, 16,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([ 4,  7, 13,  3, 18,  9, 11, 14,  2,  6, 10,  2,  8, 18,  3,  7,  5,
#          2, 10, 17, 13,  3, 10, 15, 12,  3, 11, 14,  6,  2, 10, 16,  4,  7,
#          9,  6,  4,  7, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  2,  5, 15, 12,  3, 11, 14,
#          3, 13,  8,  4, 32,  2, 35, 36, 36, 35, 33, 26, 21, 13, 10,  3, 16,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([14, 31,  7,  9, 16,  2,  8, 28,  8,  6,  2, 16, 27,  9,  5,  5,  8,
#          6,  3, 10,  2,  5, 15, 12,  3, 11, 14, 19, 15, 12,  3, 11, 14, 16,
#          9, 18, 25, 18,  5,  9,  6,  2,  5, 19,  4,  8,  9,  5,  5,  2,  5,
#          6,  3,  8,  4, 32,  2, 35, 36, 36, 35, 33, 26, 21]),
#  array([14, 31,  7,  9, 16,  2,  8, 28,  8,  6,  2, 16, 27,  9,  5,  5,  8,
#          6,  3, 10,  2,  5, 15, 12,  3, 11, 14, 19, 15, 12,  3, 11, 14, 16,
#          9, 18, 25, 18,  5,  9,  6,  2,  5, 19,  4,  8,  9,  5,  5,  2,  5,
#          6,  3,  8,  4, 32,  2, 35, 36, 36, 35, 33, 26, 21]),
#  array([14, 31,  7,  9, 16,  2,  8, 28,  8,  6,  2, 16, 27,  9,  5,  5,  8,
#          6,  3, 10,  2,  5, 15, 12,  3, 11, 14, 19, 15, 12,  3, 11, 14, 16,
#          9, 18, 25, 18,  5,  9,  6,  2,  5, 19,  4,  8,  9,  5,  5,  2,  5,
#          6,  3,  8,  4, 32,  2, 35, 36, 36, 35, 33, 26, 21]),
#  array([ 4,  7, 13,  3, 18,  9, 11, 14,  2,  6, 10,  2,  8, 18,  3,  7,  5,
#          2, 10, 21, 13,  3, 10, 15, 12,  3, 11, 14,  6,  2, 10, 16,  4,  7,
#          9,  6,  4,  7, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
#  array([ 4,  7, 13,  3, 10,  2, 11,  2,  4, 23,  2,  5, 15, 12,  3, 11, 14,
#          3, 13,  8,  4, 32,  2, 35, 36, 36, 35, 33, 26, 21, 13, 10,  3, 16,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])]

# tk.sequences_to_texts(labelled_num_seq_df['LogNumSequence'][0])
# Out[5]: 
# ['i n f o r e c e i v i n g b l o c k s r c : d e s t : UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'i n f o r e c e i v i n g b l o c k s r c : d e s t : UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'i n f o r e c e i v i n g b l o c k s r c : d e s t : UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'r o o t r a n d 8 _ t e m p o r a r y _ t a s k _ 2 0 0 8 1 1 1 0 1 0 2 4 _ 0 0 1 5 _ m _ 0 0 1 2 6 1 _ 0 p a r t - 0 1 2 6 1 .',
#  'i n f o p a c k e t r e s p o n d e r 2 f o r b l o c k t e r m i n a t i n g UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'i n f o r e c e i v e d b l o c k o f s i z e 3 5 5 3 2 4 1 f r o m UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'i n f o p a c k e t r e s p o n d e r 0 f o r b l o c k t e r m i n a t i n g UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'i n f o r e c e i v e d b l o c k o f s i z e 3 5 5 3 2 4 1 f r o m UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'k * n a m e s y s t e m . a d d s t o r e d b l o c k : b l o c k m a p u p d a t e d : i s a d d e d t o s i z e 3 5 5 3 2 4 1',
#  'k * n a m e s y s t e m . a d d s t o r e d b l o c k : b l o c k m a p u p d a t e d : i s a d d e d t o s i z e 3 5 5 3 2 4 1',
#  'k * n a m e s y s t e m . a d d s t o r e d b l o c k : b l o c k m a p u p d a t e d : i s a d d e d t o s i z e 3 5 5 3 2 4 1',
#  'i n f o p a c k e t r e s p o n d e r 1 f o r b l o c k t e r m i n a t i n g UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK',
#  'i n f o r e c e i v e d b l o c k o f s i z e 3 5 5 3 2 4 1 f r o m UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK UNK']

                 