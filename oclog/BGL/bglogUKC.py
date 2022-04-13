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
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle
import random
random.seed(123)
tf.random.set_seed(123)

class BGLog:
    
    def __init__(self, logpath='C:\ML_data\Logs', 
                 labelpath='C:\ML_data\Logs',              
                 logfilename='BGL.log',
                 train_ratio=0.8,
                 val_ratio = None,
                 test_ratio = None,
                 save_train_test_data=False,
                 seq_len=32,
                 padded_seq_len=32,
                 padded_char_len=64,
                 padding_style='post',
                 truncating='pre',
                 rm_time_stamp=True,
                 rm_msg_source=True,
                 rm_ip_address=True,
                 rm_signs_n_punctuations=True,
                 rm_white_space=True,
                 debug=False,
                 save_padded_num_sequences=False,
                 load_from_pkl=False,
                 save_dir='data',
                 pkl_file='bgl_ukc.pkl',
                 classes=['0', '1', '2', '3','4', '5', '6', ],
                 batch_size=32,                 
                ):
        self.logpath = logpath
        self.labelpath = labelpath
        self.logfilename = logfilename       
        self.logfile = os.path.join(logpath, logfilename)        
        self.train_ratio = train_ratio 
        self.seq_len = seq_len
        self.padded_seq_len = padded_seq_len
        self.padded_char_len = padded_char_len
        self.padding_style = padding_style
        self.truncating=truncating
        self.rm_time_stamp=rm_time_stamp
        self.rm_msg_source=rm_msg_source       
        self.rm_ip_address=rm_ip_address
        self.rm_signs_n_punctuations=rm_signs_n_punctuations
        self.rm_white_space=rm_white_space
        self.logs = None
        self.tk = None        
        self.labelled_txt_sequence = None       
        self.debug = debug
        self.negative_alerts = None
        self.cleaned_labelled_sequences = None
        self.padded_num_sequences=None
        self.padded_num_seq_df = None
        self.save_padded_num_sequences = save_padded_num_sequences
        self.load_from_pkl=load_from_pkl
        self.save_dir = save_dir
        self.pkl_file = pkl_file
        #self.full_pkl_path = os.path.join(self.save_dir, self.pkl_file)
        self.full_pkl_path = os.path.join(os.path.dirname(__file__), self.save_dir, self.pkl_file)
        #print('full_pkl_path', self.full_pkl_path)
        self.classes = classes
        self.train_df = None
        self.val_df = None
        self.test_df = None 
        self.ukc_df = None
        self.batch_size = batch_size
        self.train_test_categorical = None
        self.tensor_train_val_test = None
        self.ablation = 28000
        self.ukc_cnt = self.ablation
#         self.tk_path = os.path.join(self.save_dir, 'bgltk.pkl')
        self.tk_path = os.path.join(os.path.dirname(__file__), self.save_dir, 'bgltkukc.pkl')
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
                
    def get_log_lines(self):
        st_time = time.time()
        bglfile = os.path.join(self.logpath, self.logfilename)
        if self.debug is True:
            print('log file path found: ', os.path.exists(bglfile))
        with open(bglfile, 'r',  encoding='utf8') as f:
            bglraw = f.readlines()
        n_logs = len(bglraw)
        if self.debug:
            print('total number of lines in the log file:', n_logs)
            print('RAM usage: ', sys.getsizeof(bglraw) )
        self.logs = bglraw
        end_time = time.time()
        if self.debug:
            print('ending logs in memory:' , end_time - st_time)
        return bglraw  
    
    def get_labelled_txt_sequence(self):
        if self.logs is None:
            self.get_log_lines()
        alerts =  [l.split()[8] for l in self.logs]
        if self.debug:  print('alerts',len(alerts))
        unique_alerts = set(alerts)
        if self.debug: print(f'unique_alerts: {unique_alerts}')
        negative_alerts = ['FATAL', 'SEVERE', 'WARNING', 'Kill', 'FAILURE', 'ERROR']
        self.negative_alerts = negative_alerts
        sequences = [self.logs[i * self.seq_len:(i + 1) * self.seq_len] for i in range((len(self.logs)) // self.seq_len )] 
        if self.debug: print('length of list of sequence',len(sequences))
        stime = time.time()
        labelled_sequences = []
        for seq in sequences:    
            label = 'INFO'
            for s in seq:
                if s.split()[8] in negative_alerts:
                    label = s.split()[8]
            labelled_sequences.append((seq, label))          
        etime = time.time()        
#         df = pd.DataFrame(labelled_sequences, columns=['sequence', 'label'])
#         if self.debug: print(df.label.value_counts())
        self.labelled_txt_sequence = labelled_sequences
        if self.debug: 
            print(f'elapsed time: {etime - stime}')
            print('self.labelled_txt_sequence:', self.labelled_txt_sequence[0] )
        return labelled_sequences
    
    def clean_bgl(self, txt_line, clean_part_1=True, clean_part_2=True, clean_time_1=True, clean_part_4=True, clean_time_2=True, clean_part_6=True):
        part_1 = ''
        part_2 = ''
        time_1 = ''
        part_4 = ''
        time_2 = ''
        part_6 = ''
        if clean_part_1:
            part_1 = '^-\s|^\w+\s'
        if clean_part_2:
            part_2 = '\d{10}\s'
        if clean_time_1:
            time_1 = '\d{4}\.\d{2}\.\d{2}\s'
        if clean_part_4:
            part_4 = '\w\d{2}-\w\d-\w{2}-\w:\w\d{2}-\w\d{2}\s'
        if clean_time_2:
            time_2 = '\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}\s'
        if clean_part_6:
            part_6 = 'RAS'
        part_7 = '[\n]'
        signs_n_punctuations = '\]|\[|\)|\(|\=|\,|\;|\/|\{|\}[$]|[@]|[#]|[%]|[_]|[*]|[&]|[ï]|[ã]|[`]|[ð]|[-]|[\x0f]|[\x00]|[\x10]|[\x98]|[ç]|[:]|\''
        white_space = '\s'
        multiple_dots = '\.+?'
        pat =f'{part_1}|{part_2}|{time_1}|{part_4}|{time_2}|{part_6}\s|{part_7}|{signs_n_punctuations}|{white_space}|{multiple_dots}'
        s = re.sub(pat, '', txt_line)
        return s

    def get_cleaned_labelled_sequences(self):
        if self.labelled_txt_sequence is None:
            self.get_labelled_txt_sequence()
        cleaned_labelled_sequences = []
        for sequence, label in self.labelled_txt_sequence:
            cleaned_seq = []
            for line in sequence:
                cleaned_line = self.clean_bgl(line)
                cleaned_line = cleaned_line.lower()
                cleaned_seq.append(cleaned_line)
            cleaned_labelled_sequences.append((cleaned_seq, label)) 
        self.cleaned_labelled_sequences = cleaned_labelled_sequences
        return cleaned_labelled_sequences
    
    
    def get_trained_tokenizer(self):
        if self.cleaned_labelled_sequences is None:
            self.get_cleaned_labelled_sequences()
        whole_text_for_training = [line for sequence, _ in self.cleaned_labelled_sequences for line in sequence]
        if self.debug is True: print('len of whole_text_for_training',len(whole_text_for_training))
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        tk.fit_on_texts(whole_text_for_training)
        if self.debug: print('character vocabulary', len(tk.word_index))
        self.tk = tk
        return tk
    
    
    def get_padded_num_sequences(self):
        if self.cleaned_labelled_sequences is None:
            self.get_cleaned_labelled_sequences()
        if self.tk is None:
            self.get_trained_tokenizer()
        num_sequences = []
        for seq, label in self.cleaned_labelled_sequences:
            num_seq = []
            for sline in seq:
                try:        
                    num_line = self.tk.texts_to_sequences([sline])
                    padded_num_line = pad_sequences(num_line, maxlen=self.padded_char_len, 
                                                          padding=self.padding_style, truncating=self.truncating)
                    num_seq.append(padded_num_line[0])
                except Exception as e:
                    print(e)
                    print('line:', sline)   
                    break
            num_sequences.append((num_seq, label))
        self.padded_num_sequences = num_sequences
        # num_sequences[0]
        if self.debug: print(len(num_sequences))
        return num_sequences
    
    def get_padded_num_seq_df(self):
        st_time = time.time()        
        if self.load_from_pkl is True:
            if os.path.exists(self.full_pkl_path):
                with open(self.full_pkl_path, 'rb') as f:
                  self.padded_num_seq_df = pickle.load(f)
                print(f'padded_num_seq_df loaded from {self.full_pkl_path}')
            else:
                print(f'{self.full_pkl_path} not found')
            if os.path.exists(self.tk_path):
                with open(self.tk_path, 'rb') as f:
                  self.tk = pickle.load(f)
                print(f'trained tokenizer, tk, loaded from {self.tk_path}')
            else:
                print(f'{self.tk_path} not found')
        else:
            if self.padded_num_sequences is None:
                self.get_padded_num_sequences()
            numdf = pd.DataFrame(self.padded_num_sequences, columns=['seq', 'label'])
            if self.debug: print(numdf.head())
            numdf["label"].replace({"INFO": "0", "FATAL": "1", "ERROR": "2", 
                                 "WARNING": "3", "SEVERE": "4", "Kill": "5",
                                "FAILURE": "6"}, inplace=True)
            if self.debug: print(numdf.label.value_counts())
            self.padded_num_seq_df = numdf               
            if self.debug:
                end_time = time.time() 
                print(f'completed padding sequences in {end_time - st_time} sec')
            if self.save_padded_num_sequences is True:
                if os.path.exists(self.save_dir) is False:
                    print(f'{self.save_dir} does not exixt, creating it')
                    os.mkdir(self.save_dir)                 
                print(f'trying to save pickle in : {self.full_pkl_path}')
                with open(self.full_pkl_path , 'wb') as f:
                    pickle.dump(numdf, f)                
                with open(self.tk_path , 'wb') as f:
                    pickle.dump(self.tk, f)
                if os.path.exists(self.full_pkl_path): print(f'saved: {self.full_pkl_path}' )
                if os.path.exists(self.tk_path): print(f'saved: {self.tk_path}' )
        return self.padded_num_seq_df 
    
    
    def get_train_test_split_single_class(self, label=0):
        if self.padded_num_seq_df is None:
            self.get_padded_num_seq_df()
        bgldf = self.padded_num_seq_df
        train_data = None
        val_data = None
        test_data = None
        ukc_data = None ### unknown known - not present in the training data but present in the test data
        
        if self.debug: print(f'ablation set to : {self.ablation}')
        train_cnt = round(self.ablation * self.train_ratio)#### 1000 * 0.7 = 700
        remaining_cnt = round(self.ablation *(1 - self.train_ratio)) ### 1000 * (1-0.7) = 300 
        if self.val_ratio is None and self.test_ratio is None:
            val_cnt = test_cnt = remaining_cnt//2 ### 300/2 = 150 each
        else:
            val_cnt = round(self.ablation * self.val_ratio) ### 1000 * 0.2 = 200 
            test_cnt = round(self.ablation * self.test_ratio) ### 1000 * 0.1 = 100
        
        cls_data = bgldf[bgldf.label==label]
        cls_data_cnt = cls_data.count()[0] 
        
        if self.ablation <= cls_data_cnt: ### if 1000 <= 2000          
            train_data = cls_data[0:train_cnt] ### cls_data[0:700]
            val_data = cls_data[train_cnt:train_cnt+val_cnt] ### cls_data[700:(700+200)]
            test_data = cls_data[train_cnt+val_cnt:self.ablation] ### cls_data[900:1000]
        elif self.ablation > cls_data_cnt and cls_data_cnt >= train_cnt+val_cnt: ### 1000>950 and 950>(700+200)
            train_data = cls_data[0:train_cnt] ### cls_data[0:700]
            remaining_for_test = cls_data_cnt - (train_cnt+val_cnt) ### 950 - (700+200) = 50
            if remaining_for_test > 0: ### 50 > 0
                val_data = cls_data[train_cnt:train_cnt+val_cnt] ### cls_data[700:(700+200)]
                test_data = cls_data[train_cnt+val_cnt:cls_data_cnt] ### cls_data[900:950]
            else: ### cls_data_cnt = 850 or 900
                val_data = cls_data[train_cnt:cls_data_cnt] ### cls_data[700:850]
        else:
            if self.debug:
                print(f'{cls_data_cnt} data in class {label} not enough to split into train:{train_cnt} and validation:{val_cnt}')
            if cls_data_cnt < test_cnt:
                ukc_data = cls_data[0:cls_data_cnt]
            else:
                ukc_data = cls_data[0:test_cnt]
        if self.debug:    
            if train_data is not None:
                print(f'train_{label}:, {train_data.count()[0]}')
            if val_data is not None:
                print(f'val_{label}:, {val_data.count()[0]}')
            if test_data is not None:
                print(f'test_{label}:, {test_data.count()[0]}')
            if ukc_data is not None:
                print(f'ukc_{label}:, {ukc_data.count()[0]}')
        return train_data, val_data,test_data, ukc_data
    
    
    def get_train_test_multi_class(self,):
        # classes = ['NORMALBGL', 'FATALBGL', 'ERRORBGL', 'WARNINGBGL','SEVEREBGL', 'KillBGL', 'FAILUREBGL', ] 
        classes=self.classes
        if self.padded_num_seq_df is None:
            self.get_padded_num_seq_df()
        bgldf = self.padded_num_seq_df 
        train_data, val_data, test_data, ukc_data = [], [], [], []
        for class_name in self.classes:
                trdata, valdata, tsdata, ukcdata = self.get_train_test_split_single_class(label=class_name)
                if trdata is not None: train_data.append(trdata)
                if valdata is not None: val_data.append(valdata)
                if tsdata is not None: test_data.append(tsdata)
                if ukcdata is not None: ukc_data.append(ukcdata)
        self.train_df = pd.concat(train_data)
        self.val_df = pd.concat(val_data)
        self.test_df = pd.concat(test_data)
        self.ukc_df = pd.concat(ukc_data)
        ukc_num = self.ukc_df.count()[0]
        if ukc_num >= self.ukc_cnt:
            ukc_to_add = self.ukc_df[0:self.ukc_cnt]
        else:
            ukc_to_add = self.ukc_df[0:ukc_num]        
        self.test_df = pd.concat([self.test_df, ukc_to_add])
        if self.debug: 
            print(self.train_df.label.value_counts())
            print(self.val_df.label.value_counts())
            print(self.test_df.label.value_counts())
        return self.train_df, self.val_df, self.test_df  
    
    
    def get_train_test_categorical(self):
        if self.train_df is None or self.test_df is None:
            self.get_train_test_multi_class()
        x_train = list(self.train_df.seq.values)
        y_train = list(self.train_df.label.values)        
        y_train = to_categorical(y_train)
        print(y_train[:2])
        x_val = list(self.val_df.seq.values)
        y_val = list(self.val_df.label.values)
        y_val = to_categorical(y_val)
        # print(y_val[:2])
        x_test = list(self.test_df.seq.values)     
        y_test = list(self.test_df.label.values)
        unique_label_train = np.unique(self.train_df.label)
        max_label_num_train = max(unique_label_train)
        ukc_label = str(int(max_label_num_train)+1)
        self.test_df.loc[self.test_df.label > max_label_num_train, 'label' ]=ukc_label
        y_test = to_categorical(y_test)
        if self.debug:
            print('test df', self.test_df.label.value_counts())
            print(y_test[:2])
            print(y_train[80:82])
        self.train_test_categorical = x_train, y_train, x_val, y_val, x_test, y_test
        return self.train_test_categorical
    
    
    def get_tensor_train_val_test(self, ablation=None, ukc_cnt=None):
        if ablation is None: 
            ablation = self.ablation
        else:
            self.ablation = ablation
        if self.train_test_categorical is None:
            self.get_train_test_categorical()
        B = self.batch_size
        x_train, y_train, x_val, y_val, x_test, y_test = self.train_test_categorical
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.shuffle(buffer_size=y_train.shape[0]).batch(B, drop_remainder=True)
        print(train_data)
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.shuffle(buffer_size=y_val.shape[0]).batch(B, drop_remainder=True)
        print(val_data)
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_data = test_data.shuffle(buffer_size=y_test.shape[0]).batch(B, drop_remainder=True)
        print(test_data)
        if self.debug:            
            print(train_data.element_spec[0].shape[2])
            print(train_data.element_spec[1].shape[1])
        self.tensor_train_val_test = train_data, val_data, test_data
        return self.tensor_train_val_test

    
    
def get_embedding_layer(log_obj):
    tk = log_obj.tk
    vocab_size = len(tk.word_index)
    print(f'vocab_size: {vocab_size}')
    char_onehot = vocab_size
    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))
    for char, i in tk.word_index.items(): # from 1 to 51
        onehot = np.zeros(vocab_size)
        onehot[i-1] = 1
        embedding_weights.append(onehot)
    embedding_weights = np.array(embedding_weights)
    return embedding_weights, vocab_size, char_onehot    
    
        
    
    