import os
import sys
import re
import psutil
import time
import pickle 
import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools  import groupby

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle


class HDFSLogv3:
    
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
                 hdfs_rm_time_stamp=True,
                 hdfs_rm_msg_source=True,
                 hdfs_rm_blk_ids_regex=True,
                 hdfs_rm_ip_address=True,
                 hdfs_rm_signs_n_punctuations=True,
                 hdfs_rm_white_space=True,
                 debug=False
                ):
        self.logpath = logpath
        self.labelpath = labelpath
        self.logfilename = logfilename
        self.labelfilename = labelfilename
        self.logfile = os.path.join(logpath, logfilename)
        self.labelfile = os.path.join(labelpath, labelfilename)
        self.train_ratio = train_ratio
        self.split_type = split_type        
        self.padded_seq_len = padded_seq_len
        self.padded_char_len = padded_char_len
        self.padding_style = padding_style
        self.truncating=truncating
        self.hdfs_rm_time_stamp=hdfs_rm_time_stamp
        self.hdfs_rm_msg_source=hdfs_rm_msg_source
        self.hdfs_rm_blk_ids_regex=hdfs_rm_blk_ids_regex
        self.hdfs_rm_ip_address=hdfs_rm_ip_address
        self.hdfs_rm_signs_n_punctuations=hdfs_rm_signs_n_punctuations
        self.hdfs_rm_white_space=hdfs_rm_white_space
        self.logs = None
        self.tk = None        
        self.seq_of_log_texts = None
        self.seq_of_log_nums = None
        self.cleaned_logs = None
        self.blkid_to_line_to_num = None
        self.num_sequence_by_blkid = None
        self.num_seq_by_blkid_itertools = None
        self.labelled_num_seq_df = None
        self.train_test_data = None
        self.padded_train_test_data = None
        self.debug = False
    
    ### 1st - done
    def get_log_lines(self):
        st_time = time.time()
        with open(self.logfile, 'r', encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip().lower() for x in logs]
        n_logs = len(logs)
        if self.debug:
            print('total number of lines in the log file:', n_logs)
            print('RAM usage: ', sys.getsizeof(logs) )
        self.logs = logs
        end_time = time.time()
        if self.debug:
            print('ending logs in memory:' , end_time - st_time)
        return logs  
    
    
    ### 2nd - No storage
    def remove_unwanted_characters_n_words(self, txt_line, debug=False):
        # if debug:
        #     print(f'original Line: {txt_line}, original length: {len(txt_line)}' )         
        time_stamp = ''
        msg_source = ''
        blk_ids_regex = ''
        ip_address = ''
        signs_n_punctuations = ''
        white_space = ''

        if self.hdfs_rm_time_stamp:
            time_stamp = '^\d+\s\d+\s\d+' 
        if self.hdfs_rm_msg_source:
            msg_source = 'dfs\.\w+[$]\w+:|dfs\.\w+:'
        if self.hdfs_rm_blk_ids_regex:
           # blk_ids_regex = 'blk_-\d+\.?'
           blk_ids_regex = 'blk_-?\d+\.?'
        if self.hdfs_rm_ip_address:
            ip_address = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:*\d*'
        if self.hdfs_rm_signs_n_punctuations:
            signs_n_punctuations = '\]|\[|\)|\(|\=|\,|\;|\/'
        if self.hdfs_rm_white_space:
            white_space = '\s'
            
        pat = f'{time_stamp}|{msg_source}|{blk_ids_regex}|{ip_address}|{signs_n_punctuations}|{white_space}'     
        s = re.sub(pat, '', txt_line)
        # print(type(s))
        # if debug:
        #     print(f'cleaned line: {s},  cleaned length: {len(s)}')
        #     print()
        return s
    
    
    ### 3rd done 
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
        if self.debug:
            print('loaded cleaned logs with blk_id  in memory:' , end_time - st_time)
            print('RAM usage: ', sys.getsizeof(cleaned_logs) )
        return cleaned_logs
    
    
    ### 4th - no storage
    def get_cleaned_txt_without_blkid(self):
        if self.cleaned_logs is None:
            self.get_blkid_n_clean_text()
        st_time = time.time()
        cleaned_logs_witout_blkid =  [tup[1] for tup in self.cleaned_logs]
        end_time = time.time()
        if self.debug:
            print('loaded cleaned logs without blkid in memory:' , end_time - st_time)
            print('RAM usage: ', sys.getsizeof(self.cleaned_logs) )
        return cleaned_logs_witout_blkid
    
    
    ### 5th  - minor storage
    def train_char_tokenizer(self):
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        st_time = time.time()
        cleaned_logs_witout_blkid = self.get_cleaned_txt_without_blkid()
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
    def convert_char_to_numbers(self):
        if self.tk is None:
            self.train_char_tokenizer()
        if self.cleaned_logs is None:
            self.get_blkid_n_clean_text()
        if self.debug:
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
            if i % 1000000 == 0 and self.debug: 
                print('completed: ', i)
                end_time = time.time()
                print('time :' , end_time - st_time) 
        end_time = time.time()
        if self.debug:
            print('ending text to number conversion:' , end_time - st_time)        
            print('RAM usage: ', sys.getsizeof(blkid_to_line_to_num), )
        self.blkid_to_line_to_num = blkid_to_line_to_num
        return blkid_to_line_to_num
    
    
    ### 7th - done
    def get_num_sequence_by_blkid(self):
        st_time = time.time()
        if self.blkid_to_line_to_num is None:
            self.convert_char_to_numbers()
        od = OrderedDict()
        for k, v in self.blkid_to_line_to_num:
            if k not in od:
                od[k] = []
            od[k].append(v)
        self.num_sequence_by_blkid = od
        end_time = time.time()
        if self.debug:
            print('ending num_sequence_by_blkid conversion:' , end_time - st_time)        
            print('RAM usage: ', sys.getsizeof(blkid_to_line_to_num))
        return self.num_sequence_by_blkid
    
    
    ### 8th - done 
    def get_num_seq_by_blkid_itertools(self):
        st_time = time.time()
        if self.blkid_to_line_to_num is None:
            self.convert_char_to_numbers()
        result = { k : [*map(lambda v: v[1], values)]
            for k, values in groupby(sorted(self.blkid_to_line_to_num, key=lambda x: x[0]), lambda x: x[0])
            }
        self.num_seq_by_blkid_itertools = result
        end_time = time.time()
        if self.debug:
            print('ending num_sequence_by_blkid conversion:' , end_time - st_time)        
            print('RAM usage: ', sys.getsizeof(result) )
        return result
    
    
    ### 9th  this is what is required 
    def get_labelled_num_seq_df(self):
        st_time = time.time()
        if self.num_seq_by_blkid_itertools is None:
            self.get_num_seq_by_blkid_itertools()
        ######### changes: chaging the column name  as per bgl log  ##################
        labelled_num_seq_df = pd.DataFrame(list(self.num_seq_by_blkid_itertools.items()), columns=['BlockId', 'seq'])
        label_data = pd.read_csv(self.labelfile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['label'].to_dict()
        ######## changes: Don't want to convert the label to 0 or 1 , 
        ##### we want the text label and distingused as hdfs_anomaly and hdfs_normal####
        labelled_num_seq_df['label'] = labelled_num_seq_df['BlockId'].apply(
            lambda x: "hdfs_anomaly" if label_dict[x] == 'Anomaly' else "hdfs_normal")
        ##### changes: pad the sequence here itself to make all the sequence of equal length ############
        labelled_num_seq_df['seq'] = labelled_num_seq_df['BlockId'].apply(
            lambda x: pad_sequences(x, maxlen=self.padded_seq_len, 
                                              padding=self.padding_style, truncating=self.truncating))
        self.labelled_num_seq_df = labelled_num_seq_df
        end_time = time.time()
        if self.debug:
            print('ending labelled_num_seq_df conversion:' , end_time - st_time) 
            print('RAM usage: ', sys.getsizeof(labelled_num_seq_df) )
        return labelled_num_seq_df
        
        
    
    def get_train_test_data(self, ablation=0, shuffle_data=False, save_pkl=False):
        st_time = time.time()
        # (x_data, y_data) = shuffle(x_data, y_data)
        if self.labelled_num_seq_df is None:
            self.get_labelled_num_seq_df()
        x_data = self.labelled_num_seq_df['LogNumSequence'].values
        y_data = self.labelled_num_seq_df['Label'].values
        if self.split_type == 'uniform' and y_data is not None:
            pos_idx = y_data > 0
            x_pos = x_data[pos_idx]
            y_pos = y_data[pos_idx]
            x_neg = x_data[~pos_idx]
            y_neg = y_data[~pos_idx]
            # if ablation != 0 and x_pos.shape[0] >= ablation:
            #     x_pos = x_pos[0:ablation]
            #     y_pos = y_pos[0:ablation]
            #     x_neg = x_neg[0:ablation]
            #     y_neg = y_neg[0:ablation]    
            train_pos = int(self.train_ratio * x_pos.shape[0])
            train_neg = train_pos
            if ablation !=0:
                print(f'getting ablation data: {ablation}')
                train_pos = ablation
                train_neg = ablation
            x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
            y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
            if x_pos[train_pos:].shape[0] >= train_pos and x_neg[train_neg:].shape[0] >= train_neg:
                if self.debug:
                    print(f'total x_test positive: {x_pos[train_pos:].shape[0]}')
                    print(f'ablation x_test positive: {x_neg[train_neg:2*train_neg].shape[0]}')
                print(f'ablation x_test positive: {x_pos[train_pos:2*train_pos].shape[0]}')
                print(f'total x_test negative: {x_neg[train_neg:].shape[0]}')                
                x_test = np.hstack([x_pos[train_pos:2*train_pos], x_neg[train_neg:2*train_neg]])
                y_test = np.hstack([y_pos[train_pos:2*train_pos], y_neg[train_neg:2*train_neg]])
            else:
                x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
                y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        
        # Random shuffle
        if shuffle_data is True:
            indexes = shuffle(np.arange(x_train.shape[0]))
            x_train = x_train[indexes]
            if y_train is not None:
                y_train = y_train[indexes]
        print(f'{self.train_ratio} % split ratio of positve class, train:{y_train.sum()}, test:{y_test.sum()}') # 8419 8419
        self.train_test_data = x_train, y_train, x_test, y_test
        end_time = time.time()
        if self.debug:
            print('train_test_data done:' , end_time - st_time)
        # print('padded_txt_to_num shape:', padded_txt_to_num.shape) # padded_txt_to_num shape: (11175629, 320)
        # self.padded_txt_to_num = padded_txt_to_num
        if self.debug:
            print('RAM usage train_test_data: ', sys.getsizeof(self.train_test_data), )
        if save_pkl is True:
            with open('data\train.pkl' , 'wb') as f:
                pickle.dump((x_train, y_train), f)
            with open('data\test.pkl' , 'wb') as f:
                pickle.dump((x_test, y_test), f)
        return x_train, y_train, x_test, y_test
    
    
    def get_padded_train_test_data(self, save_pkl=False, hdfs_pkl_path='data', hdfs_pkl_name='hdfsv3.pkl',  ablation=0, shuffle_data=False,):
        st_time = time.time()
        if self.train_test_data is None:
            if ablation !=0:
                self.get_train_test_data(ablation=ablation)
            self.get_train_test_data()
        x_train, y_train, x_test, y_test = self.train_test_data
        if self.debug:
            for i in range(3):
                print('length of train  sequence original', len(x_train[i]))            
        padded_x_train = pad_sequences(x_train, maxlen=self.padded_seq_len, 
                                       padding=self.padding_style, truncating=self.truncating)  # 57 taken automatically
        ##(16838, 57, 230)  
        if self.debug:
            for i in range(3):
                print('length of train sequence padded', len(padded_x_train[i]))            
        padded_x_test = pad_sequences(x_test, maxlen=self.padded_seq_len, 
                                      padding=self.padding_style, truncating=self.truncating) 
        # # # (558223, 57, 230)
        if self.debug:
            for i in range(3):
                print('len of test seq after padding',len(padded_x_test[i]))
        self.padded_train_test_data = padded_x_train, y_train, padded_x_test, y_test
        end_time = time.time()
        if self.debug:
            print('padded_train_test_data done:' , end_time - st_time)        
            print('RAM usage padded_train_test_data: ', sys.getsizeof(self.padded_train_test_data), )
        if save_pkl is True:
            train_pkl = 'train_'+ hdfs_pkl_name
            test_pkl = 'test_' + hdfs_pkl_name
            if not os.path.exists(hdfs_pkl_path):
                os.mkdir(hdfs_pkl_path)                
            pkl_full_path_train = os.path.join(hdfs_pkl_path, train_pkl)
            pkl_full_path_test = os.path.join(hdfs_pkl_path, test_pkl)
            
            with open(pkl_full_path_train , 'wb') as f:
                pickle.dump((padded_x_train, y_train), f)
            with open(pkl_full_path_test , 'wb') as f:
                pickle.dump((padded_x_test, y_test), f)
        return padded_x_train, y_train, padded_x_test, y_test
    
    
    def save_hdfs_log_obj(self):
        #### clean up variable storage to reduce the obj size##
        self.logs = None ### 1
        self.cleaned_logs = None ### 2nd
        self.blkid_to_line_to_num = None ### 6th
        self.num_sequence_by_blkid = None ### 7th
        self.num_seq_by_blkid_itertools = None ### 8th          
        self.labelled_num_seq_df = None ### since this is not padded for equal seq length   
        
        ### we need a df with both char and seq padded #############
        
        
        self.padded_train_test_data = None  ##### since this will be generated based on ablation from the df
        

            

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

                 