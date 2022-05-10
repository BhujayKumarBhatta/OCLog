# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:24:31 2022

@author: Bhujay_ROG
"""

import numpy as np
import tensorflow as tf
tf.random.set_seed(123)
# from oclog.BGL.bglog import  get_embedding_layer
# from bglog import BGLog, get_embedding_layer
# bglog = BGLog(save_padded_num_sequences=False, load_from_pkl=True)
# train_test = bglog.get_tensor_train_test(ablation=1000)
# train_data, test_data = train_test

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


class LogLineEncoder(tf.keras.Model):
    def __init__(self, logobj, chars_in_line=64, num_of_conv1d=3,  
                 filters=64,
                 kernel_size=3, char_embedding_size=None):
        super().__init__()  
        self.logobj = logobj
        # self.chars_in_line = train_data.element_spec[0].shape[2]
        self.chars_in_line = chars_in_line
        self.num_of_conv1d = num_of_conv1d       
        self.filters = filters
        self.kernel_size = kernel_size        
        #TODO Done: make this varaible - bglog - 
        self.embedding_weights, self.vocab_size, self.char_onehot = get_embedding_layer(self.logobj)
        self.char_embedding_size = self.vocab_size if  char_embedding_size is None else char_embedding_size
        
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size+1,
                                    output_dim=self.char_embedding_size,
                                    #TODO: make this varaible
                                    input_length=self.chars_in_line, 
                                    weights = [self.embedding_weights],
                                    )
        self.conv1d_layers = [tf.keras.layers.Conv1D(filters=filters, 
                                                kernel_size=kernel_size, 
                                                padding='same')  
                       for _ in range(self.num_of_conv1d)]
        self.maxpool2d = tf.keras.layers.MaxPooling2D(
            pool_size=(1, self.chars_in_line)) #TODO Done: make this varaible
                  
        
    def call(self, inputs):
        x = self.embedding(inputs)
        for conv1d_layer in self.conv1d_layers:
            x = conv1d_layer(x)
        x = self.maxpool2d(x)
        #TODO: make this varaible
        tf_shape = tf.shape(inputs)  ### inputs.shape[0] does not work while model is saved.
        # x = tf.reshape(x, (inputs.shape[0], inputs.shape[1], self.filters))
        x = tf.reshape(x, (tf_shape[0], tf_shape[1], self.filters))
        return x
    

    
    
class LogSeqEncoder(tf.keras.Model):
    
    def __init__(self, line_in_seq=32, num_of_conv1d=3,  filters=64,
                 kernel_size=3, maxpool_1=True,
                 dense_neurons=16, dense_activation='relu',):
        super().__init__()
        self.line_in_seq = line_in_seq
        self.num_of_conv1d = num_of_conv1d        
        self.dense_neurons = dense_neurons
        self.filters = filters
        self.kernel_size = kernel_size
        self.maxpool_1 = maxpool_1
        self.dense_activation = dense_activation
        self.input_conv1d_layers = tf.keras.layers.Conv1D(filters=filters, 
                                                kernel_size=kernel_size, 
                                                padding='same')
        self.addl_conv1d_layers = []
        if self.num_of_conv1d > 1:            
            self.addl_conv1d_layers = [tf.keras.layers.Conv1D(filters=filters, 
                                                kernel_size=kernel_size, 
                                                padding='same')  
                       for _ in range(self.num_of_conv1d - 1)]
        self.maxpool1d = tf.keras.layers.MaxPooling1D(pool_size=(self.line_in_seq) ) #TODO done: make this varaible
        
        self.Dense = tf.keras.layers.Dense(self.dense_neurons, 
                                           activation=self.dense_activation)
       
        
    def call(self, inputs):
        # conv1d_layer = self.conv1d_layers.pop(0)
        x = self.input_conv1d_layers(inputs)
        if self.addl_conv1d_layers:
            for addl_conv1d_layer in self.addl_conv1d_layers:
                x = addl_conv1d_layer(x)
        x = self.maxpool1d(x)
        tf_shape = tf.shape(inputs)
        x = tf.reshape(x, (tf_shape[0], self.filters)) 
        x = self.Dense(x)
        return x
    
    
    
class LogClassifier(tf.keras.Model):
    
    def __init__(self,  num_classes, line_encoder, seq_encoder, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.log_line_encoder = line_encoder
        self.log_seq_encoder = seq_encoder
        self.classifier = tf.keras.layers.Dense(
            self.num_classes, activation='softmax') #TODO done: make this varaible
        # self.mymetrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(),tf.keras.metrics.Recall()] 
        # self.my_loss_tracker = tf.keras.metrics.Mean(name="my_loss")
        self.batch_features = None
        
   
    def call(self, inputs, extract_feature=False,):
        x = self.log_line_encoder(inputs)
        seq_embedding = self.log_seq_encoder(x)
        self.batch_features = seq_embedding        
        if  extract_feature:
            output = seq_embedding
        else:
            output = self.classifier(seq_embedding)
        return output   
   
  
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        print('in train step')
        x, y = data
        batch_features = x
        label_batch = y
        print('label_batch withi train step:......',label_batch)
        num_classes = self.num_classes
        embedding_size = x.shape[1]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            if self.compiled_loss and self.losses:
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                 # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            else:           
                loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
                # loss = self.custom_loss(y, y_pred)
                # print('self.batch_features', self.batch_features)
                # print('y_pred', y_pred)
                # loss = self.hvm_loss(self.batch_features, embedding_size,  y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        if self.compiled_metrics:
            self.compiled_metrics.update_state(y, y_pred)
        # else:
        #     #### learn this and complete            
        #     for m in self.mymetrics:
        #         # self.loss_tracker.update_state(loss)
        #         m.update_state(y, y_pred) 
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        # return {m.name: m.result() for m in self.mymetrics}
        
    def custom_loss(self,  y_true, y_pred):
        print('using custom loss.........................')
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=1)
    
    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [self.loss_tracker, self.mymetrics]
    
    def hvm_loss(self, batch_feature, embedding_size, label_batch, label_batch_pred):
        print('using hvm_loss .........................')  
        print('batch_feature within hvm loss', batch_feature)
        print('label_batch within hvm loss', label_batch)
        centroids = tf.zeros((self.num_classes, embedding_size))
        total_labels = tf.zeros(self.num_classes)
        for i in range(label_batch.shape[0]): # (32, 4) --> here length is 32
            label = label_batch[i] # label looks like [0 0 0 1]
            print('label', label.numpy())
            numeric_label = tf.math.argmax(label) # index position of the label = 3 , so it is actually class =3
            numeric_label = numeric_label.numpy()
            print('numeric_label', numeric_label)
            # numeric_label = np.array(tf.unstack(numeric_label))
            ##total_labels = [0 0 0 0] each col representing a class 
            ## count the number for each class
            total_labels_lst = tf.unstack(total_labels)
            total_labels_lst[numeric_label] += 1 
            total_labels = tf.stack(total_labels_lst)
            centroids_lst = tf.unstack(centroids)
            centroids_lst[numeric_label] += batch_features[i]
            centroids = tf.stack(centroids_lst)
            # self.labelled_features[numeric_label] = features[i]
            # each row index in the centroid array is a class
            # we add first identify the feature belonging to which class by the numeric_label
            # Then add all the features belonging to the class in the corresponding row of the centroid arr
        ### shape of centroids is (4, 16) whereas shape of total_labels is (1, 4)
        ### reshape the total_labels as 4,1 ==> [[0], [0], [0], [0]]==> 4 rows 
        ## so that we can divide the centroids array by the total_labels
        total_label_reshaped = tf.reshape(total_labels, (self.num_classes, 1))
        centroids /= total_label_reshaped
        pt_batch_centroids = centroids
        label_indexs = tf.math.argmax(label_batch, axis=1)
        # c = tf.gather(centroids, indices=label_indexs)
        centroid_for_features_as_per_class = tf.gather(centroids, indices=label_indexs)
        # print('centroid_for_features_as_per_class', centroid_for_features_as_per_class.shape)
        # print('first feature in batch_feature', batch_features[0], )
        euc_dis = tf.norm(batch_features - centroid_for_features_as_per_class, ord='euclidean', axis=1)
        # print('distance for the first feature', euc_dis[0], euc_dis.shape)
        loss = tf.keras.losses.categorical_crossentropy(label_batch, label_batch_pred) + tf.reduce_mean(euc_dis, )
        return loss

