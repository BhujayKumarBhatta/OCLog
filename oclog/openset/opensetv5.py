# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:24:31 2022

@author: Bhujay_ROG
"""
import numpy as np
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

class OpenSet:
    ''' 
    self.num_labels = number of classes
    self.embedding_size = number of neurons in the logits layers of the pretrained model'''
    def __init__(self, num_labels, pretrained_model, embedding_size=16, function_model=False):
#         super().__init__():
        self.pretrained_model = pretrained_model        
        self.centroids = None
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.radius = None      
        self.radius_changes = []
        self.losses = []
        self.function_model = function_model
        self.features = None
        self.pred_eudist = None
        self.pred_radius = None
        self.unknown = None
        self.best_eval_score = 0
        self.ukc_label = 999
    
    def train(self, data_train, lr_rate=0.05, epochs=1, wait_patient=3, optimizer='adam'):
        lossfunction = BoundaryLoss(num_labels=self.num_labels)    
        ### ### why is it needed ? 
        #self.radius = tf.nn.softplus(lossfunction.theta)
        ### ### calculate centroid  after each ephochs after a fresh training
        #self.centroids = self.centroids_cal(data_train) 
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
            
            
        wait, best_radius, best_centroids = 0, None, None 
        for epoch in range(epochs):
            ### ### calculate centroid after one more round of triaing
            self.pretrained_model.fit(data_train)
            self.centroids = self.centroids_cal(data_train)  
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
            _, _, eval_score = self.evaluate(data_train, debug=False)           
            print(f'epoch: {epoch+1}/{epochs}, train_loss: {loss.numpy()}, eval_score: {eval_score}' )
            if eval_score > self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_radius = self.radius
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= wait_patient:
                    break        
        self.radius = best_radius
        self.centroids = best_centroids
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
                # each row index in the centroid array is a class
                # we add first identify the feature belonging to which class by the numeric_label
                # Then add all the features belonging to the class in the corresponding row of the centroid arr
        ### shape of centroids is (4, 16) whereas shape of total_labels is (1, 4)
        ### reshape the total_labels as 4,1 ==> [[0], [0], [0], [0]]==> 4 rows 
        ## so that we can divide the centroids array by the total_labels
        total_label_reshaped = tf.reshape(total_labels, (self.num_labels, 1))
        centroids /= total_label_reshaped
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
    
    def evaluate(self, data, debug=True, zero_div=1, ukc_label=None):
        # if hasattr(data, ukc_label)
        if ukc_label is None:
            ukc_label = self.ukc_label
        else:
            self.ukc_label = ukc_label
        total_preds, total_labels = [], []
        for batch in data:
            logseq_batch, label_batch = batch
            features_batch = self.get_pretrained_features(logseq_batch)
            preds_np = self.openpredict(features_batch, debug=False)            
            label_indexs = tf.math.argmax(label_batch, axis=1)
            label_index_np = label_indexs.numpy()
            total_preds.append(preds_np)
            total_labels.append(label_index_np)
        y_pred = np.array(total_preds).flatten().tolist()
        y_true = np.array(total_labels).flatten().tolist()
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
        return y_true, y_pred, f1_weighted
    
    def plot_radius_chages(self):
        narr = np.array([elem.numpy() for elem in self.radius_changes])
        tnsr = tf.convert_to_tensor(narr)        
        tpose = tf.transpose(tnsr)
        losses = [elem.numpy() for elem in self.losses]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1) # 1 row 2 column , first plot        
        # fig = sns.lineplot(data=[tpose.numpy()[0][0], 
        #                    tpose.numpy()[0][1],
        #                   tpose.numpy()[0][2],
        #                   tpose.numpy()[0][3]])
        fig = sns.lineplot(data=[tpose.numpy()[0][i] for i in range(self.num_labels)])
        fig.set_xlabel("Epochs")
        fig.set_ylabel("Radius")
        plt.subplot(1, 2, 2) # # 1 row 2 column , 2nd plot
        fig2 = sns.lineplot(data=[losses])
        fig2.set_xlabel("Epochs")
        fig2.set_ylabel("Loss")
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