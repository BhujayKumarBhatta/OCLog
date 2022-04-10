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
    
    def train(self, data_train, lr_rate=0.05, epochs=1):
        lossfunction = BoundaryLoss(num_labels=self.num_labels)       
        self.radius = tf.nn.softplus(lossfunction.theta)
        self.centroids = self.centroids_cal(data_train)        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate) # does it take criterion_boundary.parameters() ??
        wait,best_delta, best_centroids = 0, None, None        
        for epoch in range(epochs):
            # self.pretrained_model.fit(data_train)
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
#             loss = tf.reshape(loss, loss.shape[0])
            print(f'epoch: {epoch+1}/{epochs}, train_loss: {loss.numpy()}' )
            
            
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

    def openpredict(self, features):
        logits = euclidean_metric(features, self.centroids)
        ####original line in pytorch ###probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        smax = tf.nn.softmax(logits, )
        preds = tf.math.argmax(smax, axis=1)
        probs = tf.reduce_max(smax, 1)            
        #######euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        pred_centroids = tf.gather(self.centroids, indices=preds)
        euc_dis = tf.norm(features - pred_centroids, ord='euclidean', axis=1)
        print('euc_dis:',euc_dis)
        pred_radius = tf.gather(self.radius, indices=preds)
        pred_radius = tf.reshape(pred_radius, pred_radius.shape[0], )        
        print('pred_radius:',pred_radius)
        #####preds[euc_dis >= self.delta[preds]] = data.unseen_token_id
        unknowns = euc_dis >= pred_radius
#         preds[unknowns] = 0000
        print('unknowns:', unknowns)
        return preds
    
    def evaluate(self, data):
        for batch in tqdm(data):
            pass
            
        
    
    def plot_radius_chages(self):
        narr = np.array([elem.numpy() for elem in self.radius_changes])
        tnsr = tf.convert_to_tensor(narr)        
        tpose = tf.transpose(tnsr)
        losses = [elem.numpy() for elem in self.losses]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1) # 1 row 2 column , first plot
        fig = sns.lineplot(data=[tpose.numpy()[0][0], 
                           tpose.numpy()[0][1],
                          tpose.numpy()[0][2],
                          tpose.numpy()[0][3]])
        fig.set_xlabel("Epochs")
        fig.set_ylabel("Radius")
        plt.subplot(1, 2, 2) # # 1 row 2 column , 2nd plot
        fig2 = sns.lineplot(data=[losses])
        fig2.set_xlabel("Epochs")
        fig2.set_ylabel("Loss")
        plt.show()
        