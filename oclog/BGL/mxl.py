
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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from oclog.hdfs.hdflogv3 import HDFSLogv3
sys.path.append('C:\\Users\\Bhujay_ROG\\MyDev\\OCLog\\oclog\hdfs')

from sklearn.utils import shuffle
import random
random.seed(123)
tf.random.set_seed(123)

class MixedLog:
        
    def __init__(self, **kwargs):        
        self.debug = kwargs.get('debug', False)
        self.logpath = kwargs.get('logpath', 'C:\ML_data\Logs')
        self.labelpath = kwargs.get('labelpath', 'C:\ML_data\Logs' )
        self.logfilename = kwargs.get('logfilename', 'BGL.log')
        ##########################
        self.ablation = kwargs.get('ablation', 1000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.val_ratio = kwargs.get('val_ratio', None)
        self.test_ratio = kwargs.get('test_ratio', None)
       
        
        self.ukc_cnt = kwargs.get('ukc_cnt', self.ablation)
        self.clean_part_1 = kwargs.get('clean_part_1', True)
        self.clean_part_2 = kwargs.get('clean_part_2', True)
        self.clean_time_1 = kwargs.get('clean_time_1', True)
        self.clean_part_4 = kwargs.get('clean_part_4', True)
        self.clean_time_2 = kwargs.get('clean_time_2', True)
        self.clean_part_6 = kwargs.get('clean_part_6', True)
        ####### HDFS realted 
        self.hdfs_rm_time_stamp = kwargs.get('rm_time_stamp', True)
        self.hdfs_rm_msg_source = kwargs.get('rm_msg_source', True)
        self.hdfs_rm_ip_address = kwargs.get('rm_ip_address', True)
        self.hdfs_rm_signs_n_punctuations = kwargs.get('rm_signs_n_punctuations', True)
        self.hdfs_rm_white_space = kwargs.get('rm_white_space', True)
        
        self.log_meta_status = kwargs.get('log_meta_status', 'no_meta')
        if self.log_meta_status == 'time_ip':
            ### For BGL
            self.clean_part_1 = False
            self.clean_part_2 = False
            self.clean_time_1 = False
            self.clean_part_4 = False
            self.clean_time_2 = False
            self.clean_part_6 = False
            #### for HDFS
            self.hdfs_rm_time_stamp = False
            self.hdfs_rm_msg_source = False
            self.hdfs_rm_ip_address = False
            
        ##########################
        self.seq_len = kwargs.get('seq_len', 32)
        self.padded_seq_len = kwargs.get('padded_seq_len', 32)
        self.padded_char_len = kwargs.get('padded_char_len', 64)
        self.padding_style = kwargs.get('padding_style', 'post')
        self.truncating = kwargs.get('truncating', 'pre')
        ###### bgl pk file will be saved based on the parameter of seq, char len and meta status, similar naming convention as HDFS
        default_file_name = 'bgl_ukc_' + str(self.padded_seq_len) + '_' + str(self.padded_char_len) + '_' + self.log_meta_status
        print('saved file name will be: ', default_file_name)
        default_bgl_pkl_filename = default_file_name + '.pkl'
        default_bgl_tk_filename = default_file_name + '_tk' + '.pkl'
        self.save_dir = kwargs.get('save_dir', 'data')
        self.pkl_file = kwargs.get('pkl_file', default_bgl_pkl_filename)
        self.tk_file = kwargs.get('tk_file', default_bgl_tk_filename)
        self.save_padded_num_sequences = kwargs.get('save_padded_num_sequences', False)
        self.save_train_test_data = kwargs.get('save_train_test_data', False)
        self.load_from_pkl = kwargs.get('load_from_pkl', False)
       
        ########################################
        
        self.logfile = os.path.join(self.logpath, self.logfilename)   
        self.full_pkl_path = os.path.join(os.path.dirname(__file__), self.save_dir, self.pkl_file)
        self.tk_path = os.path.join(os.path.dirname(__file__), self.save_dir, self.tk_file)
        
        self.hdfs_obj_save_path = kwargs.get('hdfs_obj_save_path', 'C:\\Users\\Bhujay_ROG\\MyDev\OCLog\\oclog\\hdfs\\data')
        self.hdfs_obj_name = kwargs.get('hdfs_obj_name', None)
        self.mixed_logs = kwargs.get('mixed_logs', False)
        # self.classes = kwargs.get('classes', ['INFO', 'FATAL', 'ERROR', 'WARNING','SEVERE', 'Kill', 'FAILURE', ]) ## 7 classes
        self.classes = kwargs.get('classes', ['0', '1', '2', '3','4', '5', '6', ])
        # if self.mixed_logs:
        #     self.classes = kwargs.get('classes', ['0', '1', '2', '3','4', ])
            # self.classes = kwargs.get('classes', ['INFO', 'hdfs_anomaly', 'hdfs_normal', 'FATAL', 'ERROR', 
                                                  # 'WARNING','SEVERE', 'Kill', 'FAILURE', ]) ## 9 classes
        
        self.logs = None
        self.tk = None        
        self.labelled_txt_sequence = None
        self.negative_alerts = None
        self.cleaned_labelled_sequences = None
        self.padded_num_sequences=None
        self.padded_num_seq_df = None 
        self.train_df = None
        self.val_df = None
        self.test_df = None 
        self.ukc_df = None        
        self.train_test_categorical = None
        self.tensor_train_val_test = None
        ### allocate the last class as the ukc class by default
        # self.designated_ukc_cls = kwargs.get('designated_ukc_cls', self.classes[len(self.classes) - 1] ) 
        self.designated_ukc_cls = None
        self.label_ukc_as = None
        
        
                
    def get_log_lines(self, **kwargs):
        st_time = time.time()
        bglfile = os.path.join(self.logpath, self.logfilename)
        if self.debug is True:
            print('bgl log file path found: ', os.path.exists(bglfile))
        with open(bglfile, 'r',  encoding='utf8') as f:
            bglraw = f.readlines()
        n_logs = len(bglraw)
        if self.debug:
            print('total number of lines in the bgl log file:', n_logs)
            print('RAM usage: ', sys.getsizeof(bglraw) )
        self.logs = bglraw
        end_time = time.time()
        if self.debug:
            print('ending logs in memory:' , end_time - st_time)
        return bglraw 
    
    
    def get_labelled_txt_sequence(self, **kwargs):
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
        # if self.debug: print(df.label.value_counts())
        self.labelled_txt_sequence = labelled_sequences
        if self.debug: 
            print(f'elapsed time: {etime - stime}')
            #print('self.labelled_txt_sequence:', self.labelled_txt_sequence[0] )
        return labelled_sequences
    
    
    def clean_bgl(self, txt_line, **kwargs):
        part_1 = ''
        part_2 = ''
        time_1 = ''
        part_4 = ''
        time_2 = ''
        part_6 = ''
        if self.clean_part_1:
            part_1 = '^-\s|^\w+\s'
        if self.clean_part_2:
            part_2 = '\d{10}\s'
        if self.clean_time_1:
            time_1 = '\d{4}\.\d{2}\.\d{2}\s'
        if self.clean_part_4:
            part_4 = '\w\d{2}-\w\d-\w{2}-\w:\w\d{2}-\w\d{2}\s'
        if self.clean_time_2:
            time_2 = '\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}\s'
        if self.clean_part_6:
            part_6 = 'RAS'
        part_7 = '[\n]'
        signs_n_punctuations = '\]|\[|\)|\(|\=|\,|\;|\/|\{|\}[$]|[@]|[#]|[%]|[_]|[*]|[&]|[ï]|[ã]|[`]|[ð]|[-]|[\x0f]|[\x00]|[\x10]|[\x98]|[ç]|[:]|\''
        white_space = '\s'
        multiple_dots = '\.+?'
        pat =f'{part_1}|{part_2}|{time_1}|{part_4}|{time_2}|{part_6}\s|{part_7}|{signs_n_punctuations}|{white_space}|{multiple_dots}'
        s = re.sub(pat, '', txt_line)
        return s
    

    def get_cleaned_labelled_sequences(self, **kwargs):
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
    
    
    def get_trained_tokenizer(self, **kwargs):
        if self.cleaned_labelled_sequences is None:
            self.get_cleaned_labelled_sequences()
        whole_text_for_training = [line for sequence, _ in self.cleaned_labelled_sequences for line in sequence]
        if self.debug is True: print('len of whole_text_for_training',len(whole_text_for_training))
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        tk.fit_on_texts(whole_text_for_training)
        if self.debug: print('character vocabulary', len(tk.word_index))
        self.tk = tk
        return tk
    
    
    def get_padded_num_sequences(self, **kwargs):
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
        if self.debug: print('len of numseq: ',len(num_sequences))
        return num_sequences
    
    
    def get_padded_num_seq_df(self, **kwargs):
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
            if self.debug: print('how the bgl padded_num_seq_df looks:\n', numdf.head(2))
            ########################################################
            numdf["label"].replace({"INFO": "0", "FATAL": "1", "ERROR": "2", 
                                 "WARNING": "3", "SEVERE": "4", "Kill": "5",
                                "FAILURE": "6"}, inplace=True)
            map_dict = {"INFO": "0", "FATAL": "1", "ERROR": "2", 
                                 "WARNING": "3", "SEVERE": "4", "Kill": "5",
                                "FAILURE": "6"}
            print('class label to number mapping for bgl:' , map_dict)
            ############################################################
            if self.debug: print('count of each class in bgl padded_num_seq_df:\n', numdf.label.value_counts().to_dict())
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
    
    
    ######################## MIX LOG #########################
    ################################################
       
    def get_hdfs_df(self, **kwargs):
        # load according to the seq, char and meta setting  or generate
        # data\\hdfsobj_32_176_time_ip.pkl
        #### HDFS param and BGL param matches
        # self.padded_seq_len = kwargs.get('padded_seq_len', 32)
        # self.padded_char_len = kwargs.get('padded_char_len', 64)
        
        #### try to loacate existing hdfs object saved file based on parameters  
        if self.log_meta_status == 'no_meta':
            hdfs_rm_time_stamp = True
            hdfs_rm_ip_address = True
        else:
            hdfs_rm_time_stamp = False
            hdfs_rm_ip_address = False           
        default_file_name = 'hdfsobj_' + str(self.padded_seq_len) + '_' + str(self.padded_char_len) +'_' + self.log_meta_status + '.pkl' 
        if not self.hdfs_obj_name:
            hdfs_obj_name = default_file_name
        else:
            hdfs_obj_name = self.hdfs_obj_name
        hdfs_obj_full_name_path = os.path.join(self.hdfs_obj_save_path, hdfs_obj_name)
        
        ##### if the file does not exist generate the hdfs log object and save it 
        if not os.path.exists(hdfs_obj_full_name_path):
            if self.debug:
                print(f'pre-existing file: {hdfs_obj_full_name_path} for saved hdfs object not found, generating one')
            hlog = HDFSLogv3(debug=self.debug )
            hlog.get_tensor_train_val_test(padded_seq_len=self.padded_seq_len, 
                                                               padded_char_len=self.padded_char_len, 
                                                               hdfs_rm_time_stamp=hdfs_rm_time_stamp, 
                                                               hdfs_rm_ip_address=hdfs_rm_ip_address, 
                                                               train_ratio=0.8, )
            fname = hlog.save_hdfs_log_obj(hdfs_obj_save_path=default_file_path)
            hdfs_obj_full_name_path = os.path.abspath(fname) ### to be converted to full file name
            print(f'hdfs log save as : {fname}')
        else:
            print(f'found existing hdfs saved object from {hdfs_obj_full_name_path}')
        
        #### hdfs file must exist now either generated or pre-existing
        with open(hdfs_obj_full_name_path, 'rb') as f:
            hdfslogs = pickle.load(f)
        lebeled_num_seq_df_epn = hdfslogs.lebeled_num_seq_df_epn
        if self.debug: print('classes in hdfs df, lebeled_num_seq_df_epn: \n', lebeled_num_seq_df_epn.label.value_counts().to_dict())
        return lebeled_num_seq_df_epn
    
    
    def get_mix_log_df(self, **kwargs): 
        # numdf["label"].replace({"INFO": "0", "FATAL": "1", "ERROR": "2", 
        #                          "WARNING": "3", "SEVERE": "4", "Kill": "5",
        #                         "FAILURE": "6"}, inplace=True)
        if self.debug: print('############## you want to mix log from HDFS, let me try that ##################### ')
        if self.debug: print('#### I will change the existing label of BGL to accomodate hdfs label #####')
        bgl_df = self.padded_num_seq_df
        if bgl_df is None:
            bgl_df = self.get_padded_num_seq_df()
        selected_labels = (bgl_df.label == '0') | (bgl_df.label == '1') | (bgl_df.label == '2') | (bgl_df.label == '3') | (bgl_df.label == '4') 
        bgl_df = bgl_df[selected_labels]
        if self.debug: print('taking few selected classes from original bgl: ', bgl_df.label.value_counts().to_dict())
        bgl_df["label"].replace({"0": "0", "1": "1", "2": "4", "3": "5","4": "6"}, inplace=True)
        if self.debug: print('changed bgl classes: ', bgl_df.label.value_counts().to_dict())
        hdfs_df = self.get_hdfs_df(**kwargs)
        #############################################################
        hdfs_class_map = {"hdfs_anomaly": "2", "hdfs_normal": "3"}
        hdfs_df["label"].replace(hdfs_class_map, inplace=True)
        print('hdfs class map: ', hdfs_class_map)
        if self.debug: print('changed hdfs classes: ', hdfs_df.label.value_counts().to_dict())
        #########################################################
        mixed_df = pd.concat([bgl_df, hdfs_df])
        if self.debug: print('Merged the bgl and hdfs level as the mixed df: ', mixed_df.label.value_counts().to_dict())
        return mixed_df
   ############################################## Mixed Log end ################### 

    
    def get_train_test_split_single_class(self, bgldf, label=0, **kwargs):
        # if self.padded_num_seq_df is None:
        #     self.get_padded_num_seq_df()
        # bgldf = self.padded_num_seq_df
        train_data = None
        val_data = None
        test_data = None
        ukc_data = None ### unknown known - not present in the training data but present in the test data        
        
        train_cnt = round(self.ablation * self.train_ratio)#### 1000 * 0.7 = 700
        remaining_cnt = round(self.ablation *(1 - self.train_ratio)) ### 1000 * (1-0.7) = 300 
        if self.val_ratio is None and self.test_ratio is None:
            val_cnt = test_cnt = remaining_cnt//2 ### 300/2 = 150 each
        else:
            val_cnt = round(self.ablation * self.val_ratio) ### 1000 * 0.2 = 200 
            test_cnt = round(self.ablation * self.test_ratio) ### 1000 * 0.1 = 100
        
        cls_data = bgldf[bgldf.label==label]
        cls_data_cnt = cls_data.count()[0]
        # cls_unique_label = int(np.unique(cls_data.label)[0])
        #if self.debug: print('cls_unique_label', cls_unique_label)
        # if self.designated_ukc_cls == cls_unique_label:
        
        if str(self.designated_ukc_cls) == str(label):
            if cls_data_cnt < test_cnt:
                ukc_data = cls_data[0:cls_data_cnt]
            else:
                ukc_data = cls_data[0:test_cnt]
            print(f'class {label} is added as ukc')
        else:
            if self.debug: print(f'class_{label} not a desgnated ukc, hence adding in train, val and test')
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
                    print(f'{cls_data_cnt} data in class {label} not enough to split into train:{train_cnt} and validation:{val_cnt}, adding the entire data as temp  ukc bucket from where later actual ukc can be taken')
                # if self.designated_ukc_cls is None:    
                if cls_data_cnt < test_cnt:
                    ukc_data = cls_data[0:cls_data_cnt]
                else:
                    ukc_data = cls_data[0:test_cnt]
        # if self.debug:    
        if train_data is not None:
            print(f'train_{label}:, {train_data.count()[0]}')
        if val_data is not None:
            print(f'val_{label}:, {val_data.count()[0]}')
        if test_data is not None:
            print(f'test_{label}:, {test_data.count()[0]}')
        if ukc_data is not None:
            print(f'ukc_{label}:, {ukc_data.count()[0]}')
        return train_data, val_data,test_data, ukc_data
    
    
    def get_train_test_multi_class(self, **kwargs):
        # classes = ['NORMALBGL', 'FATALBGL', 'ERRORBGL', 'WARNINGBGL','SEVEREBGL', 'KillBGL', 'FAILUREBGL', ] 
        classes=self.classes
        if self.debug: print(f'ablation set to : {self.ablation}')
        if self.debug: print('parameter value for designated_ukc_cls within multiclass: ', self.designated_ukc_cls)
        if self.padded_num_seq_df is None:
            self.get_padded_num_seq_df(**kwargs)
        bgldf = self.padded_num_seq_df
        if self.mixed_logs:
            bgldf = self.get_mix_log_df(**kwargs)
        train_data, val_data, test_data, ukc_data = [], [], [], []
        for class_name in self.classes:
                trdata, valdata, tsdata, ukcdata = self.get_train_test_split_single_class(bgldf, label=class_name, **kwargs)
                if trdata is not None: train_data.append(trdata)
                if valdata is not None: val_data.append(valdata)
                if tsdata is not None: test_data.append(tsdata)
                if ukcdata is not None: ukc_data.append(ukcdata)
        self.train_df = pd.concat(train_data)
        self.val_df = pd.concat(val_data)
        self.test_df = pd.concat(test_data)
        if ukc_data:
            self.ukc_df = pd.concat(ukc_data)
            ukc_num = self.ukc_df.count()[0]
            print('total data in  temp ukc bucket:', ukc_num)
            if ukc_num >= self.ukc_cnt:
                ukc_to_add = self.ukc_df[0:self.ukc_cnt]
                if self.debug: print('slicing ukc bucket as per ukc_cnt parameter value:', self.ukc_cnt)
            else:
                ukc_to_add = self.ukc_df[0:ukc_num]
                if self.debug: print(f'you wanted {self.ukc_cnt} ukc data but that much is not there.  adding {ukc_num} to ukc data')
            print('ukc_to_add df : ', ukc_to_add.label.value_counts().to_dict())
            self.test_df = pd.concat([self.test_df, ukc_to_add])
        if self.debug: 
            print('train classes:', self.train_df.label.value_counts().to_dict())
            print('val classes:', self.val_df.label.value_counts().to_dict())
            print('test classes:', self.test_df.label.value_counts().to_dict())
        return self.train_df, self.val_df, self.test_df  
    
    
    def get_train_test_categorical(self, **kwargs):
        # if self.train_df is None or self.test_df is None:
        ### when called with different param by get_tensor_train_val_test this should be always recallled. 
        self.get_train_test_multi_class(**kwargs)
        x_train = list(self.train_df.seq.values)
        y_train = list(self.train_df.label.values)        
        y_train = to_categorical(y_train)
        
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
            print('test df', self.test_df.label.value_counts().to_dict())
            print('some example of labels:')
            print('first two train label', y_train[:2])
            print('first two test label', y_test[:2])
            # print(y_train[80:82])
        self.train_test_categorical = x_train, y_train, x_val, y_val, x_test, y_test
        return self.train_test_categorical
        
        # if self.debug:
        #     # print('test df', self.test_df.label.value_counts())
        #     print('some example of labels:')
        #     print('first two label in train:\n', y_train[:2])
        #     print('first two label in tests:\n', y_test[:2])
        #     print('80th and 82nd train label:\n', y_train[80:82])
        # self.train_test_categorical = x_train, y_train, x_val, y_val, x_test, y_test
        # return self.train_test_categorical
    
    
    def get_tensor_train_val_test(self, **kwargs):
        self.ablation = kwargs.get('ablation', self.ablation)
        self.designated_ukc_cls = kwargs.get('designated_ukc_cls', self.designated_ukc_cls)
        self.ukc_cnt = kwargs.get('ukc_cnt', self.ukc_cnt)
        self.train_ratio = kwargs.get('train_ratio', self.train_ratio)
        self.val_ratio = kwargs.get('train_ratio', self.val_ratio)
        self.test_ratio = kwargs.get('train_ratio', self.test_ratio)
        
        # if self.train_test_categorical is None:
        # we need to regenerate the slice from df
        self.get_train_test_categorical(**kwargs)
        B = self.batch_size
        x_train, y_train, x_val, y_val, x_test, y_test = self.train_test_categorical
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.shuffle(buffer_size=y_train.shape[0]).batch(B, drop_remainder=True)
        
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.shuffle(buffer_size=y_val.shape[0]).batch(B, drop_remainder=True)
        
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_data = test_data.shuffle(buffer_size=y_test.shape[0]).batch(B, drop_remainder=True)
        
        if self.debug: 
            # if self.debug: print(self.padded_num_seq_df.label.value_counts())
            print('train_data',train_data)
            print('val_data', val_data)
            print('test_data', test_data)
            print('char in lines, train_data.element_spec[0].shape[2]',train_data.element_spec[0].shape[2])
            print('num classes, train_data.element_spec[1].shape[1]: ',train_data.element_spec[1].shape[1])            
            print('length of val_data:',len(val_data))
        print('length of train_data - (num_seq_per_cls * num_class)// batch size:', len(train_data))
        self.tensor_train_val_test = train_data, val_data, test_data
        return self.tensor_train_val_test

    
    
def get_embedding_layer(log_obj):
    tk = log_obj.tk
    vocab_size = len(tk.word_index)
    if self.debug: print(f'vocab_size: {vocab_size}')
    char_onehot = vocab_size
    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))
    for char, i in tk.word_index.items(): # from 1 to 51
        onehot = np.zeros(vocab_size)
        onehot[i-1] = 1
        embedding_weights.append(onehot)
    embedding_weights = np.array(embedding_weights)
    return embedding_weights, vocab_size, char_onehot    
    
        
    
    