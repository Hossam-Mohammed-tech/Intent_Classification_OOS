# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:32:05 2023

@author: hossa
"""
import csv
import sys
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import operator
from keras.layers import *
from keras.callbacks import *

def read_tsv_file(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        texts = []
        labels = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if line[1] != '':
                texts.append(line[0])
                if line[1] == 'oos':
                    labels.append('oos\n')#(line[1])
                else:
                    labels.append(line[1])
        return texts, labels

def indexes(iterable, obj):
    return (index for index, elem in enumerate(iterable) if elem == obj)

def construct_synthetic_outliers(train_labels, use_embed_train, tsdae_embed_train, synthetic_outliers_train_size, known_label_list, oos_intent_str):
    
    synthetic_outliers_use_embed_train  = []
    synthetic_outliers_tsdae_embed_train  = []
        
    use_embed_out_1 = []
    use_embed_out_2 = []
        
    tsdae_embed_out_1 = []
    tsdae_embed_out_2 = []
        
    for k in range(synthetic_outliers_train_size):
        label_1, label_2 = np.random.choice(known_label_list, 2, replace=False)

        #print('label_1: ', label_1)
        #print('label_2: ', label_2)
        
        indices_1 = indexes(train_labels, label_1)
        indices_2 = indexes(train_labels, label_2)
        
        indices_1 = list(indices_1)
        indices_2 = list(indices_2)
        
        #print('indices_1: ', indices_1)
        #print('indices_2: ', indices_2)
        
        index_1 = random.choice(indices_1)
        index_2 = random.choice(indices_2)
    
        use_embed_out_1.append(use_embed_train[index_1])
        use_embed_out_2.append(use_embed_train[index_2])
        
        tsdae_embed_out_1.append(tsdae_embed_train[index_1])
        tsdae_embed_out_2.append(tsdae_embed_train[index_2])
    
    print('len(use_embed_out_1): ', len(use_embed_out_1))
    print('len(use_embed_out_2): ', len(use_embed_out_2))
    
    print('len(tsdae_embed_out_1): ', len(tsdae_embed_out_1))
    print('len(tsdae_embed_out_2): ', len(tsdae_embed_out_2))
    
    for i in range(len(use_embed_out_1)):
        theta = np.random.uniform(0, 1)

        use_embed_text_out_sum = (theta * use_embed_out_1[i]) + ((1 - theta) * use_embed_out_2[i])
        tsdae_embed_text_out_sum = (theta * tsdae_embed_out_1[i]) + ((1 - theta) * tsdae_embed_out_2[i])

        synthetic_outliers_use_embed_train.append(use_embed_text_out_sum)
        synthetic_outliers_tsdae_embed_train.append(tsdae_embed_text_out_sum)
    
    print('len(synthetic_outliers_use_embed_train): ', len(synthetic_outliers_use_embed_train))
    print('len(synthetic_outliers_use_embed_train[0]): ', len(synthetic_outliers_use_embed_train[0]))
    print('len(synthetic_outliers_tsdae_embed_train): ', len(synthetic_outliers_tsdae_embed_train))
    print('len(synthetic_outliers_tsdae_embed_train[0]): ', len(synthetic_outliers_tsdae_embed_train[0]))
    
    return synthetic_outliers_use_embed_train, synthetic_outliers_tsdae_embed_train


def construction_outliers(train_labels, train_text, synthetic_outliers_train_size, known_label_list_all, all_neg_text, open_domain_train_size, use_embed, tsdae_embed, oos_intent_str):
    outliers_use_embed_train = []
    outliers_tsdae_embed_train = []
    outliers_train_labels = []
        
    selected_train_text_1 = []
    selected_train_text_2 = []
    
    #print('train_labels: ', train_labels)
    
    print('known_label_list_all: ', known_label_list_all)
    
    indices = np.where(known_label_list_all==oos_intent_str)
    known_label_list = np.delete(known_label_list_all, indices)
    
    print('modified_known_label_list: ', known_label_list)
    
    for i in range(synthetic_outliers_train_size):
        label_1, label_2 = np.random.choice(known_label_list, 2, replace=False)

        #print('label_1: ', label_1)
        #print('label_2: ', label_2)
        
        indices_1 = indexes(train_labels, label_1)
        indices_2 = indexes(train_labels, label_2)
        
        indices_1 = list(indices_1)
        indices_2 = list(indices_2)
        
        #print('indices_1: ', indices_1)
        #print('indices_2: ', indices_2)
        
        index_1 = random.choice(indices_1)
        index_2 = random.choice(indices_2)
        
        #index_1 = np.random.choice(list(indices_1))
        #index_2 = np.random.choice(list(indices_2))
        #print('index_1: ', index_1)
        #print('index_2: ', index_2)
        
        #print('train_text[index_1]: ', train_text[index_1])
        #print('train_text[index_2]: ', train_text[index_2])
        #print('train_labels[index_1]: ', train_labels[index_1])
        #print('train_labels[index_2]: ', train_labels[index_2])
        
        selected_train_text_1.append(train_text[index_1])
        selected_train_text_2.append(train_text[index_2])
    
    print('len(selected_train_text_1): ', len(selected_train_text_1))
    print('len(selected_train_text_2): ', len(selected_train_text_2))
    
    selected_synthetic_outliers_use_embed_train, selected_synthetic_outliers_tsdae_embed_train = compute_synthetic_outliers(selected_train_text_1, selected_train_text_2, use_embed, tsdae_embed)
    
    open_domain_outliers = np.random.choice(all_neg_text, size=open_domain_train_size, replace=False)
    
    use_embed_open_domain_outliers = np.array(use_embed(np.array(open_domain_outliers))['outputs'])
    print('len(use_embed_open_domain_outliers): ', len(use_embed_open_domain_outliers))

    tsdae_embed_open_domain_outliers = np.array(tsdae_embed.encode(open_domain_outliers))
    tsdae_embed_open_domain_outliers = tf.convert_to_tensor(tsdae_embed_open_domain_outliers, dtype=tf.float32)
    
    outliers_use_embed_train = selected_synthetic_outliers_use_embed_train
    outliers_tsdae_embed_train = selected_synthetic_outliers_tsdae_embed_train
    
    for i in range(len(use_embed_open_domain_outliers)):
        outliers_use_embed_train.append(use_embed_open_domain_outliers[i])
        outliers_tsdae_embed_train.append(tsdae_embed_open_domain_outliers[i])
        
    for i in range(len(outliers_tsdae_embed_train)):
        outliers_train_labels.append(oos_intent_str)
    
    print('len(outliers_use_embed_train): ', len(outliers_use_embed_train))
    print('len(outliers_use_embed_train[1]): ', len(outliers_use_embed_train[1]))
    print('len(outliers_tsdae_embed_train): ', len(outliers_tsdae_embed_train))
    print('len(outliers_tsdae_embed_train[1]): ', len(outliers_tsdae_embed_train[1]))
    print('len(outliers_train_labels): ', len(outliers_train_labels))
    
    outliers_use_embed_train = np.array(outliers_use_embed_train)
    outliers_tsdae_embed_train = np.array(outliers_tsdae_embed_train)

    return outliers_use_embed_train, outliers_tsdae_embed_train, outliers_train_labels
    

def compute_synthetic_outliers(train_text_1, train_text_2, use_embed, tsdae_embed):
    synthetic_outliers_use_embed_train  = []
    synthetic_outliers_tsdae_embed_train  = []
    
    use_embed_text_out_1 = np.array(use_embed(np.array(train_text_1))['outputs'])
    print('len(use_embed_text_out_1): ', len(use_embed_text_out_1))
    
    use_embed_text_out_2 = np.array(use_embed(np.array(train_text_2))['outputs'])
    print('len(use_embed_text_out_2): ', len(use_embed_text_out_2))

    tsdae_embed_text_out_1 = np.array(tsdae_embed.encode(train_text_1))
    tsdae_embed_text_out_1 = tf.convert_to_tensor(tsdae_embed_text_out_1, dtype=tf.float32)
    
    tsdae_embed_text_out_2 = np.array(tsdae_embed.encode(train_text_2))
    tsdae_embed_text_out_2 = tf.convert_to_tensor(tsdae_embed_text_out_2, dtype=tf.float32)

    for i in range(len(train_text_1)):
        theta = np.random.uniform(0, 1)

        use_embed_text_out_sum = (theta * use_embed_text_out_1[i]) + ((1 - theta) * use_embed_text_out_2[i])
        tsdae_embed_text_out_sum = (theta * tsdae_embed_text_out_1[i]) + ((1 - theta) * tsdae_embed_text_out_2[i])

        synthetic_outliers_use_embed_train.append(use_embed_text_out_sum)
        synthetic_outliers_tsdae_embed_train.append(tsdae_embed_text_out_sum)
    
    print('len(synthetic_outliers_use_embed_train): ', len(synthetic_outliers_use_embed_train))
    print('len(synthetic_outliers_use_embed_train[0]): ', len(synthetic_outliers_use_embed_train[0]))
    print('len(synthetic_outliers_tsdae_embed_train): ', len(synthetic_outliers_tsdae_embed_train))
    print('len(synthetic_outliers_tsdae_embed_train[0]): ', len(synthetic_outliers_tsdae_embed_train[0]))
    
    return synthetic_outliers_use_embed_train, synthetic_outliers_tsdae_embed_train


### Compute Stat Functions
def compute_metric_indomain_oos_dev_1_thr(dev_labels, model_output_dev, unique_y_train, oos_intent_str, THRESHOLDS, f_test, percentage_max_accuracy_arr):

    thr1_acc_scores = []
    
    print('------------ compute_metric_indomain_oos_dev_single_1_thr_2_modified -------------')
    print('len(model_output_dev): ', len(model_output_dev))
            
    for j in range(len(THRESHOLDS)):   
        thr1_preds = []
        
        Thr1 = THRESHOLDS[j]
        print('Thr1 = ', Thr1)
    
        for i in range(len(model_output_dev)):
            model_pred = model_output_dev[i]
            max_model_index = np.argmax(model_pred)
            max_model_prop = model_pred[max_model_index]
            
            if(max_model_prop >= Thr1):
                pred = unique_y_train[max_model_index]
            else:    
                pred = oos_intent_str
                                       
            thr1_preds.append(pred)
           
            if(pred != dev_labels[i]):
                print('pred: ', pred, '  -  actual: ', dev_labels[i], '  -  max_model_prop: ', max_model_prop,  
                      '  -  max_model_pred: ', unique_y_train[max_model_index])
                print('------------------------------------')
      
        print('len(thr1_preds:)', len(thr1_preds))
        print('len(dev_labels:)', len(dev_labels))
        
        acc_score_1 = accuracy_score(thr1_preds, dev_labels)
        acc_score_1 = acc_score_1*100
        
        thr1_acc_scores.append(acc_score_1)
  
        print('\n Dev_acc_score_1:', acc_score_1)
        print('\n------------------------------------\n')
        
    print('THRESHOLDS: ', THRESHOLDS)
    print('thr1_acc_scores: ', thr1_acc_scores)
      
    print('-----------  Max accuracy  -----------')
    max_value_thr1 = max(thr1_acc_scores)
    max_index_thr1 = [index for index in range(len(thr1_acc_scores)) if thr1_acc_scores[index] == max_value_thr1]
    max_index_thr1 = max_index_thr1[-1]

    print('max_value_thr1: ', max_value_thr1)
    print('max_index_thr1: ', max_index_thr1)
    
    best_thr1 = THRESHOLDS[max_index_thr1]
    print('max_value_thr1: ', max_value_thr1)
    print('best_thr1: ', best_thr1)
    
    best_thr1_arr = []
    max_value_thr1_arr = []
    count = 4
    print('-----------  Fixed percentages of the max accuracy  -----------')
    for i in range(len(percentage_max_accuracy_arr)):
        percentage_max_accuracy = percentage_max_accuracy_arr[i]
        
        print('count: ', count)
        if(percentage_max_accuracy == -1):
            updated_max_value = thr1_acc_scores[max_index_thr1 - count]
            best_thr1 = THRESHOLDS[max_index_thr1 - count]
            count = count - 1
        else:
            if((max_index_thr1 + count) < len(thr1_acc_scores)):
                updated_max_value = thr1_acc_scores[max_index_thr1 + count]
                best_thr1 = THRESHOLDS[max_index_thr1 + count]
                count = count + 1

                #updated_max_value = max_value_thr1 * percentage_max_accuracy
                #updated_max_index_thr1 = [index for index in range(len(thr1_acc_scores)) if thr1_acc_scores[index] >= updated_max_value]

                #print('updated_max_value: ', updated_max_value)
                #print('updated_max_index_thr1: ', updated_max_index_thr1)

                #updated_max_index_thr1 = updated_max_index_thr1[-1]
                #print('updated_max_index_thr1_last: ', updated_max_index_thr1)

                #best_thr1 = THRESHOLDS[updated_max_index_thr1]

            print('updated_max_value: ', updated_max_value)
            print('best_thr1: ', best_thr1)

            best_thr1_arr.append(best_thr1)
            max_value_thr1_arr.append(updated_max_value)
    
    best_thr1_arr.append(0.75)
    max_value_thr1_arr.append(99.99)
    
    best_thr1_arr.append(0.80)
    max_value_thr1_arr.append(99.99)
    
    best_thr1_arr.append(0.85)
    max_value_thr1_arr.append(99.99)
    
    best_thr1_arr.append(0.90)
    max_value_thr1_arr.append(99.99)    
    
    best_thr1_arr.append(0.95)
    max_value_thr1_arr.append(99.99)
    
    return best_thr1_arr, max_value_thr1_arr


def domain_preds_function_test_1_thr(modified_test_labels, model_output_test, best_thr1, unique_y_train, oos_intent_str, f_test):
     
    all_preds = []
    
    print('----------------- domain_preds_function_test_1_thr_2 ---------------------')
    print('len(model_output_test): ', len(model_output_test))
    
    for i in range(len(model_output_test)):        
        model_pred = model_output_test[i]            
        #max_model_index, max_model_prop = max(enumerate(model_pred), key=operator.itemgetter(1))
        max_model_index = np.argmax(model_pred)
        max_model_prop = model_pred[max_model_index]
        
        if(max_model_prop >= best_thr1):
            #pred = unique_modified_test_labels[max_model_index]
            pred = unique_y_train[max_model_index]
            #print('pred_best_thr1: ', pred)
        else:
            pred = oos_intent_str

        all_preds.append(pred)
        
        if(pred != modified_test_labels[i]):
            print('pred: ', pred, '  -  actual: ', modified_test_labels[i], 
                  '  -  max_model_prop: ', max_model_prop, 
                  '  -  max_model_pred: ', unique_y_train[max_model_index])    
            print('--------------------------------------')
    
    print('len(all_preds): ', len(all_preds))
    print('len(modified_test_labels):', len(modified_test_labels))
    
    return all_preds

# prepare target
def prepare_targets(y_train, y_dev):
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_enc = label_encoder.transform(y_train)
    y_dev_enc = label_encoder.transform(y_dev)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train_enc = y_train_enc.reshape(len(y_train_enc), 1)
    y_train_onehot_encoded = onehot_encoder.fit_transform(y_train_enc)
        
    y_dev_enc = y_dev_enc.reshape(len(y_dev_enc), 1)
    y_dev_onehot_encoded = onehot_encoder.fit_transform(y_dev_enc)
    
    unique_y_train = np.unique(y_train)
    
    return y_train_onehot_encoded, y_dev_onehot_encoded, unique_y_train


class EarlyStoppingValAcc(Callback):
    def __init__(self, monitor='val_accuracy', patience=0, file=''):
        super(Callback, self).__init__()
        self.patience = patience
        self.file = file
        self.best_weights = None
        self.monitor = monitor
        self.best_score = 0
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        #self.best_v_loss = np.Inf
        self.best_score = 0
        
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
              
        if (current > self.best_score):
            #self.best_v_loss = v_loss
            self.best_score = current    
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                if self.file is not None:
                    self.file.write('best_model_val_accuracy: {}'.format(self.best_score))
                    self.file.write('\n')
        
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        #return self.best_score, self.best_thr


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr