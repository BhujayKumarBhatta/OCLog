# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:24:31 2022

@author: Bhujay_ROG
"""
import os
from openpyxl import Workbook
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
tf.random.set_seed(123)
# from BGL.bglog import BGLog, get_embedding_layer
# from pretraining import LogLineEncoder, LogSeqEncoder, LogClassifier
from oclog.openset.boundary_loss import euclidean_metric, BoundaryLoss
from tqdm import trange, tqdm, tnrange
# from time import sleep
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import sklearn.metrics as m
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class OpenSet:
    ''' 
    self.num_labels = number of classes
    self.embedding_size = number of neurons in the logits layers of the pretrained model'''
    def __init__(self, num_labels, pretrained_model, embedding_size=16, function_model=False, pretrain_hist=None, 
                ukc_label=9):
#         super().__init__():
        self.pretrained_model = pretrained_model        
        self.centroids = None
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.radius = None      
        self.radius_changes = []
        self.losses = []
        self.f1_tr_lst = []
        self.f1_val_lst = []
        self.function_model = function_model
        self.features = None
        self.total_features = []
        self.total_preds = []
        # self.euclidean_metric = defaultdict(list)
        self.pred_eudist = None
        self.pred_radius = None
        self.unknown = None
        self.best_eval_score = 0
        self.ukc_label = ukc_label
        self.pretrain_hist = pretrain_hist
        self.figsize = (20, 12)
        self.epoch = 0
        self.best_train_score = 0
        self.best_val_score = 0
        self.perplexity = 200
        self.early_exaggeration = 12
        self.random_state = 123 
        self.learning_rate = 80
        self.feature_pic_size = 20
        self.centroid_pic_size = 200
        self.tf_random_seed = 1234
        self.ptmodel_name = 'ptmodel'
    
    def train(self, data_train, data_val=None, data_test=None, lr_rate=0.05, epochs=1, wait_patience=3, optimizer='adam',
             pretrain_hist=None, figsize=(20, 10), perplexity=None, early_exaggeration=None, random_state=None, learning_rate=None,
                          feature_pic_size=None, centroid_pic_size=None):
        lossfunction = BoundaryLoss(num_labels=self.num_labels)    
        ### ### why is it needed ? 
        #self.radius = tf.nn.softplus(lossfunction.theta)
        ### ### calculate centroid  after each ephochs after a fresh training
        self.centroids = self.centroids_cal(data_train)       
        if optimizer == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr_rate) # does it take criterion_boundary.parameters() ??
        elif optimizer == 'sgd':
             optimizer = tf.keras.optimizers.SGD(learning_rate=lr_rate)
        elif optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_rate)
        elif optimizer == 'adam':
             optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)
        else:
            print(f'unknown optimizer {optimizer}. assigning default as adam')
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)    
        
        if perplexity: self.perplexity = perplexity
        if early_exaggeration: self.early_exaggeration = early_exaggeration
        if learning_rate: self.learning_rate = learning_rate
        if random_state: self.random_state = random_state
        if feature_pic_size: self.feature_pic_size = feature_pic_size
        if centroid_pic_size: self.centroid_pic_size = centroid_pic_size   
      
        self.pretrain_hist = pretrain_hist    
            
        wait, best_radius, best_centroids = 0, None, None 
        for epoch in range(epochs):
            ### ### calculate centroid after one more round of triaing
            # this is increasing the loss instead of decreasing in each epoch
            #self.pretrained_model.fit(data_train)
            #self.centroids = self.centroids_cal(data_train)  
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0            
            for batch in tqdm(data_train):
                logseq_batch, label_batch = batch ## (32, 32, 64), (32, 4)
                batch_loss, self.radius = self.train_step(lossfunction, 
                                                     logseq_batch, label_batch, optimizer)
                tr_loss += batch_loss
                nb_tr_steps += 1                
            self.radius_changes.append(self.radius)
            loss = tr_loss / nb_tr_steps
            self.losses.append(tr_loss)
            _, _, eval_score_train, _ = self.evaluate(data_train, debug=False, store_features=True)
            self.f1_tr_lst.append(round(eval_score_train, 4))
            if data_val:
                _, _, eval_score_val, _ = self.evaluate(data_val, debug=False)
                self.f1_val_lst.append(round(eval_score_val, 4))
                print(f'epoch: {epoch+1}/{epochs}, train_loss: {loss.numpy()}, F1_train: {eval_score_train} '
                      f'F1_val: {eval_score_val}')
            else:
                print(f'epoch: {epoch+1}/{epochs}, train_loss: {loss.numpy()}, F1_train: {eval_score_train}')            
            
            if (eval_score_train > self.best_train_score) or (eval_score_val > self.best_val_score):
                wait = 0
                if eval_score_train > self.best_train_score:                
                    self.best_train_score = eval_score_train
                if data_val and eval_score_val > self.best_val_score:
                    self.best_val_score = eval_score_val
                best_radius = self.radius
                best_centroids = self.centroids                
            else:    
                wait += 1
                if eval_score_train <= self.best_train_score:
                    print(f'train score not improving  going to wait state {wait}')
                if eval_score_val <= self.best_val_score:
                    print(f'val score not improving  going to wait state {wait}')                
                if wait >= wait_patience:                    
                    break
            self.epoch = epoch
        self.radius = best_radius
        self.centroids = best_centroids        
         _, _, f1_weighted, f_measure = self.evaluate(data_train, ukc_label=self.ukc_label, store_features=True)
        self.plot_radius_chages()
        return self.losses, self.radius_changes
    
            
    def train_step(self, Lfunction, logseq_batch, label_batch, optimizer):       
        with tf.GradientTape() as tape:                
            #features_batch = self.model(logseq_batch, extract_feature=True)
            features_batch = self.get_pretrained_features(logseq_batch)
            loss, self.radius = Lfunction(features_batch, self.centroids, label_batch)        
            gradients = tape.gradient(loss, [self.radius])
            optimizer.apply_gradients(zip(gradients, [self.radius]))
        return loss, self.radius
       
        
    def get_pretrained_features(self, logseq_batch):
        if self.function_model is True:
            penultimate_layer = self.pretrained_model.layers[len(self.pretrained_model.layers) -2]
#             features = penultimate_layer.output
        else:
            features = self.pretrained_model(logseq_batch, extract_feature=True)
        self.features = features
        return self.features
        
    def centroids_cal(self, data):        
        centroids = tf.zeros((self.num_labels, self.embedding_size))
        total_labels = tf.zeros(self.num_labels)
        for batch in data:
            logseq_batch, label_batch = batch
            ## (32, 32, 64), (32, 4)
            features = self.get_pretrained_features(logseq_batch)
            ## (32, 16) features - 32 sequence of line each haaving 64 characrers
            ## produces a feaure vector of dimension 16. 
            for i in range(len(label_batch)): # (32, 4) --> here length is 32
                label = label_batch[i] # label looks like [0 0 0 1]
                numeric_label = np.argmax(label) # index position of the label = 3 , so it is actually class =3
                ##total_labels = [0 0 0 0] each col representing a class 
                ## count the number for each class
                total_labels_lst = tf.unstack(total_labels)
                total_labels_lst[numeric_label] += 1 
                total_labels = tf.stack(total_labels_lst)
                centroids_lst = tf.unstack(centroids)
                centroids_lst[numeric_label] += features[i]
                centroids = tf.stack(centroids_lst)
                # self.labelled_features[numeric_label] = features[i]
                # each row index in the centroid array is a class
                # we add first identify the feature belonging to which class by the numeric_label
                # Then add all the features belonging to the class in the corresponding row of the centroid arr
        ### shape of centroids is (4, 16) whereas shape of total_labels is (1, 4)
        ### reshape the total_labels as 4,1 ==> [[0], [0], [0], [0]]==> 4 rows 
        ## so that we can divide the centroids array by the total_labels
        total_label_reshaped = tf.reshape(total_labels, (self.num_labels, 1))
        centroids /= total_label_reshaped
        # TODO: the centroid for a class is a scalar or vector ?
        return centroids  

    def openpredict(self, features, debug=True):
        logits = euclidean_metric(features, self.centroids)
        ####original line in pytorch ###probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        smax = tf.nn.softmax(logits, )
        preds = tf.math.argmax(smax, axis=1)
        probs = tf.reduce_max(smax, 1)            
        #######euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        pred_centroids = tf.gather(self.centroids, indices=preds)
        euc_dis = tf.norm(features - pred_centroids, ord='euclidean', axis=1)        
        pred_radius = tf.gather(self.radius, indices=preds)
        pred_radius = tf.reshape(pred_radius, pred_radius.shape[0], )
        #####preds[euc_dis >= self.delta[preds]] = data.unseen_token_id
        unknown_filter = euc_dis >= pred_radius
        #convert to numpy since tensor is immutable
        unknown_filter_np = unknown_filter.numpy()
        preds_np = preds.numpy()
        preds_np[unknown_filter_np]=self.ukc_label  
        if debug:
            print('euc_dis:',euc_dis)
            print('pred_radius:',pred_radius)
            print(f'predictions with ukc_label={self.ukc_label}', preds_np)
        return preds_np
    
    def evaluate(self, data, debug=True, zero_div=1, ukc_label=None, store_features=False):
        # if hasattr(data, ukc_label)
        if ukc_label is None:
            ukc_label = self.ukc_label
        else:
            self.ukc_label = ukc_label
        total_features, total_preds, total_labels = [], [], []
        num_samples = 0
        for batch in data:
            logseq_batch, label_batch = batch
            features_batch = self.get_pretrained_features(logseq_batch)
            preds_np = self.openpredict(features_batch, debug=False)            
            label_indexs = tf.math.argmax(label_batch, axis=1)
            label_index_np = label_indexs.numpy()
            total_preds.append(preds_np)
            total_labels.append(label_index_np)
            total_features.append(features_batch)
            num_samples += 1
        y_pred = np.array(total_preds).flatten().tolist()        
        y_true = np.array(total_labels).flatten().tolist()
        if store_features:
            self.total_preds = y_pred
            total_features = np.array(total_features)
            total_features = np.reshape(total_features, ((total_features.shape[0] * total_features.shape[1]), total_features.shape[2])    ) 
            self.total_features = total_features
        cm = confusion_matrix(y_true, y_pred)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=zero_div)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=zero_div, )
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=zero_div, )
        cls_report = m.classification_report(y_true, y_pred)
        f_measure = self.F_measure(cm)
        if debug:
            print(cm)
            print(acc)
            print(f'f1_weighted: {f1_weighted}, f1_macro: {f1_macro}, '
                  f'f1_micro: {f1_micro}, f_measure: {f_measure}')
            print(cls_report)
            self.plot_centroids(total_features, total_preds)
        return y_true, y_pred, f1_weighted, f_measure
    
    def plot_radius_chages(self, perplexity=None, early_exaggeration=None, random_state=None, learning_rate=None,
                          feature_pic_size=None, centroid_pic_size=None):
        
        if perplexity: self.perplexity = perplexity
        if early_exaggeration: self.early_exaggeration = early_exaggeration
        if learning_rate: self.learning_rate = learning_rate
        if random_state: self.random_state = random_state
        if feature_pic_size: self.feature_pic_size = feature_pic_size
        if centroid_pic_size: self.centroid_pic_size = centroid_pic_size   
        
        if self.pretrain_hist:
            pre_scores = self.pretrain_hist.history
            # pre_scores = self.pretrain_hist.history.copy()
            # score_keys = ['accuracy', 'precision', 'recall', 'val_accuracy', 'val_precision', 'val_recall']
            pre_scores = {k:pre_scores[k] for k in pre_scores.keys() if 'loss' not in k }
            pre_scores_df = pd.DataFrame(pre_scores)
            # print(pre_scores_df)
            # pre_losses = self.pretrain_hist.history.copy()
            pre_losses = self.pretrain_hist.history
            # loss_keys = ['loss', 'val_loss']
            pre_losses = {k:pre_losses[k] for k in pre_losses.keys() if 'loss' in k }
            pre_losses_df = pd.DataFrame(pre_losses)
            # print(pre_losses_df)
        narr = np.array([elem.numpy() for elem in self.radius_changes])
        tnsr = tf.convert_to_tensor(narr)        
        tpose = tf.transpose(tnsr)
        radius = [tpose.numpy()[0][i] for i in range(self.num_labels)]
        losses = [elem.numpy() for elem in self.losses]
        f1_tr = np.array(self.f1_tr_lst) * 100
        val_tr = np.array(self.f1_val_lst) * 100
        # print(radius)
        f1_scores = [f1_tr, val_tr]
        f1_scores = pd.DataFrame({'train':f1_tr, 'val': val_tr})        
        plt.figure(figsize=self.figsize)
        if self.pretrain_hist:
            plt.subplot(3, 2, 1) ### 2 rows 2 column , first plot
            fig1 = sns.lineplot(data=pre_scores_df, )
            plt.legend(loc=0)
            fig1.set_ylabel("pre-training Scores")
            fig1.set_xlabel("Pre-training Epochs")
            plt.subplot(3, 2, 2) ### 1st row 2 column , 2nd plot
            fig2 = sns.lineplot(data=pre_losses_df)
            fig2.set_xlabel("Pre-training Epochs")
            fig2.set_ylabel("pre-training Loss")
            plt.subplot(3, 2, 3) # 2nd row 1st column , 3rd plot
        else:
            plt.subplot(2, 2, 1) # 1 row 2 column , first plot
        fig3a = sns.lineplot(data=radius)
        plt.legend(loc=0)
        fig3a.set_xlabel("Epochs")
        fig3a.set_ylabel("Radius")
        ax2 = plt.twinx()
        fig3b=sns.lineplot(data=f1_scores, color="purple", ax=ax2,)
        fig3b.set_ylabel("F1 score")
        ax2.legend(loc=1)
        if self.pretrain_hist:
            plt.subplot(3, 2, 4) # 2nd row 2nd column , 4th plot
        else:
            plt.subplot(2, 2, 2) # # 1 row 2 column , 2nd plot
        fig4 = sns.lineplot(data=[losses])
        fig4.set_xlabel("Epochs")
        fig4.set_ylabel("Loss")
        if self.pretrain_hist:
            ax5 = plt.subplot(3, 2, 5) # 2nd row 2nd column , 4th plot
        else:
            ax5 = plt.subplot(2, 2, 3) # # 1 row 2 column , 2nd plot
        if len(self.total_features) > 0:
            a = np.array(self.total_features)
            c = self.centroids.numpy()
            p = np.array(self.total_preds)
            tsne = TSNE(perplexity=self.perplexity, early_exaggeration=self.early_exaggeration, 
                        random_state=self.random_state, learning_rate=self.learning_rate)
            tout = tsne.fit_transform(a)
            cout = tsne.fit_transform(c)
            m_scaler = MinMaxScaler()
            s_scalar = StandardScaler()
            # scaled_tout = m_scaler.fit_transform(tout)
            # scaled_cout = m_scaler.fit_transform(cout)
            scaled_tout = s_scalar.fit_transform(tout)
            scaled_cout = s_scalar.fit_transform(cout)
            fig5 = ax5.scatter(scaled_tout[:, 0], scaled_tout[:, -1], c=p, s=self.feature_pic_size, cmap='tab10', )
            legend1 = ax5.legend(*fig5.legend_elements(),
                                loc="upper left", title="Classes",  bbox_to_anchor=(1.05, 1))
            ax5.add_artist(legend1)
            cmap_1 = fig5.get_cmap().colors
            ccolor = np.array([cmap_1[i] for i in  range(len(c))])
            ax5.scatter(scaled_cout[:, 0], scaled_cout[:, -1],  s=self.centroid_pic_size, c=ccolor, cmap='tab10', marker=r'd', edgecolors= 'k')
            ax5.set_xlabel("class features and their centroids")
        plt.show()
        
        
    def F_measure(self, cm):
        idx = 0
        rs, ps, fs = [], [], []
        n_class = cm.shape[0]
        for idx in range(n_class):
            TP = cm[idx][idx]
            r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
            p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
            f = 2 * r * p / (r + p) if (r + p) != 0 else 0
            rs.append(r * 100)
            ps.append(p * 100)
            fs.append(f * 100)
        f = np.mean(fs).round(4)
        f_seen = np.mean(fs[:-1]).round(4)
        f_unseen = round(fs[-1], 4)
        result = {}
        result['Known'] = f_seen
        result['Open'] = f_unseen
        result['F1-score'] = f
        return result
    
    
    def train_ptmodel(self, bglog=None, train_data=None, val_data=None, ptmodel_name='ptmodel', pt_wait=3, 
                               pre_training_checkpoint=True, chars_in_line=64, line_in_seq=32,
                               pt_optimizer='adam', pt_loss='categorical_crossentropy', pretrain_epochs=5):        
        tf.random.set_seed(self.tf_random_seed)
        if bglog is None:
            train_data, val_data,  test_data, bglog = self.get_bgdata()
        line_encoder = LogLineEncoder(bglog, chars_in_line=chars_in_line)
        logSeqencer =  LogSeqEncoder(line_in_seq=line_in_seq, dense_neurons=embedding_size)
        ptmodel = LogClassifier(line_encoder=line_encoder, seq_encoder=logSeqencer, num_classes=num_classes)
        ptmodel.compile(optimizer=pt_optimizer, loss=pt_loss,
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
          
        if ptmodel_name:
            self.ptmodel_name = ptmodel_name 
        print(datetime.datetime.now())
        print('starting to create {} automatically'.format(self.model_name))
        curr_dt_time = datetime.datetime.now()
        model_name = self.ptmodel_name + '_init_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}.h5' 
        monitor_metric = 'categorical_accuracy' 
        if val_data is not None:            
            filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'
        monitor_metric = 'val_loss'
        checkpoint = ModelCheckpoint(filepath, monitor=monitor_metric, verbose=1, 
                                     save_best_only=True, save_weights_only=False, 
                                     mode='auto', period=1)

        LR = ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=2, cooldown=1, verbose=1)
        earlystop = EarlyStopping(monitor=monitor_metric, min_delta=0, patience=pt_wait, verbose=1)
        if pre_training_checkpoint:
            callbacks_list = [checkpoint, LR, earlystop] 
        print('staring pre trining')
        hist = ptmodel.fit(train_data, validation_data=val_data, epochs=pretrain_epochs,
                          callbacks=callbacks_list,) 
        
        self.plot_pretrain_result(hist)        
        return ptmodel, hist
    
    
    def plot_pretrain_result(self, hist):
        pre_scores = hist.history
        pre_scores = {k:pre_scores[k] for k in pre_scores.keys() if 'loss' not in k }
        pre_scores_df = pd.DataFrame(pre_scores)       
        pre_losses = hist.history        
        pre_losses = {k:pre_losses[k] for k in pre_losses.keys() if 'loss' in k }
        pre_losses_df = pd.DataFrame(pre_losses)   
        plt.figure(figsize=self.figsize)
        plt.subplot(1, 2, 1) ### 1 rows 2 column , first plot
        fig1 = sns.lineplot(data=pre_scores_df, )
        plt.legend(loc=0)
        fig1.set_ylabel("pre-training Scores")
        fig1.set_xlabel("Pre-training Epochs")
        plt.subplot(1, 2, 2) ### 1st row 2 column , 2nd plot
        fig2 = sns.lineplot(data=pre_losses_df)
        fig2.set_xlabel("Pre-training Epochs")
        fig2.set_ylabel("pre-training Loss")
        plt.show()
    
    
    def get_bgdata(self, bglog_model, logpath, padded_char_len=64, ablation=5000, designated_ukc_cls=3,
                    num_classes=2, embedding_size=128,lr_rate=3, pt_optimizer='sgd',
                    pretrain_epochs=3, debug=False,  batch_size=32, train_ratio=0.8,
                    clean_part_1=False, clean_part_2=False, clean_time_1=False, clean_part_4=False, 
                    clean_time_2=False,clean_part_6=False, tk_file='bgl_tk_176.pkl', pkl_file='bgl_ukc_176.pkl',
                    save_padded_num_sequences=False, load_from_pkl=False,  ):
        bglog = bglog_model(logpath=logpath,save_padded_num_sequences=save_padded_num_sequences, debug=debug, 
                      load_from_pkl=load_from_pkl, tk_file=tk_file, pkl_file=pkl_file, 
                      batch_size=batch_size, train_ratio=train_ratio,
                      padded_char_len=padded_char_len,  
                      clean_part_1=clean_part_1, clean_part_2=clean_part_2,
                      clean_time_1=clean_time_1, clean_part_4=clean_part_4, 
                      clean_time_2=clean_time_2,clean_part_6=clean_part_6,
                      )
        train_test = bglog.get_tensor_train_val_test(ablation=2500, designated_ukc_cls=designated_ukc_cls)
        train_data, val_data,  test_data = train_test
        return train_data, val_data,  test_data, bglog
    
    
    def plot_centroids(self, total_features, total_preds,  row=1, col=1, fig=1, perplexity=None,
                       early_exaggeration=None, random_state=None, learning_rate=None,
                      feature_pic_size=None, centroid_pic_size=None):
        
        if perplexity: self.perplexity = perplexity
        if early_exaggeration: self.early_exaggeration = early_exaggeration
        if learning_rate: self.learning_rate = learning_rate
        if random_state: self.random_state = random_state
        if feature_pic_size: self.feature_pic_size = feature_pic_size
        if centroid_pic_size: self.centroid_pic_size = centroid_pic_size
               
        
        a = np.array(total_features)
        c = self.centroids.numpy()
        p = np.array(total_preds)
        tsne = TSNE(perplexity=self.perplexity, early_exaggeration=self.early_exaggeration, 
                    random_state=self.random_state, learning_rate=self.learning_rate)
        tout = tsne.fit_transform(a)
        cout = tsne.fit_transform(c)
        m_scaler = MinMaxScaler()
        s_scalar = StandardScaler()
        # scaled_tout = m_scaler.fit_transform(tout)
        # scaled_cout = m_scaler.fit_transform(cout)
        scaled_tout = s_scalar.fit_transform(tout)
        scaled_cout = s_scalar.fit_transform(cout)
        
        ax5 = plt.subplot(row, col, fig) # # 1 row 2 column , 2nd plot
        fig5 = ax5.scatter(scaled_tout[:, 0], scaled_tout[:, -1], c=p, s=self.feature_pic_size, cmap='tab10', )
        legend1 = ax5.legend(*fig5.legend_elements(),
                            loc="upper left", title="Classes",  bbox_to_anchor=(1.05, 1))
        ax5.add_artist(legend1)
        cmap_1 = fig5.get_cmap().colors
        ccolor = np.array([cmap_1[i] for i in  range(len(c))])
        ax5.scatter(scaled_cout[:, 0], scaled_cout[:, -1],  s=self.centroid_pic_size, c=ccolor, cmap='tab10', marker=r'd', edgecolors= 'k')
        ax5.set_xlabel("class features and their centroids")
    plt.show()