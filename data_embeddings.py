# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:15:12 2023

@author: hossa
"""

import numpy as np
import tensorflow as tf

## Prepare the data embeddings
def train_data_embeddings(train_text, train_labels, use_embed_train_oos, tsdae_embed_train_oos, oos_train_labels, use_embed, tsdae_embed, oos_intent_str):
    use_embed_train = []
    tsdae_embed_train = []
    train_labels_all = []
    
    train_labels_all = train_labels
    use_embed_train_1 = use_embed(np.array(train_text))['outputs']
    tsdae_embed_train_1 = tsdae_embed.encode(train_text)
    
    for i in range(len(use_embed_train_1)):
        use_embed_train.append(use_embed_train_1[i])
        tsdae_embed_train.append(tsdae_embed_train_1[i])
    
    print('len(use_embed_train): ', len(use_embed_train))
    print('len(tsdae_embed_train): ', len(tsdae_embed_train))
    print('len(train_labels): ', len(train_labels))
    
    for i in range(len(oos_train_labels)):
        use_embed_train.append(use_embed_train_oos[i])
        tsdae_embed_train.append(tsdae_embed_train_oos[i])
        train_labels_all.append(oos_train_labels[i])
    
    print('len(use_embed_train): ', len(use_embed_train))
    print('len(tsdae_embed_train): ', len(tsdae_embed_train))
    print('len(train_labels_all): ', len(train_labels_all))
    
    use_embed_train = {'input_1': np.array(use_embed_train)}
    
    tsdae_embed_train = tf.convert_to_tensor(np.array(tsdae_embed_train), dtype=tf.float32)
    tsdae_embed_train = {'input_2': tsdae_embed_train}
    
    use_tsdae_embed = [use_embed_train, tsdae_embed_train]
    
    unique_train_label = np.unique(train_labels_all)
    #unique_train_label = np.array(train_labels_all.unique().tolist())
    print('len(unique_train_label): ', len(unique_train_label))
    print('unique_train_label:', unique_train_label)
    
    number_of_classes = len(np.unique(train_labels_all))
    print('number_of_classes:', number_of_classes)
    
    oos_index = np.where(unique_train_label == oos_intent_str)[0][0]
    print('oos_index_train:', oos_index)
    
    return use_tsdae_embed, train_labels_all, oos_index, number_of_classes
    

def evaluate_data_embeddings(texts, labels, use_embed, tsdae_embed, oos_intent_str):
    
    use_embed_arr = []
    tsdae_embed_arr = []
        
    use_embed_arr = use_embed(np.array(texts))['outputs']
    tsdae_embed_arr = tsdae_embed.encode(texts)
     
    print('len(use_embed_arr): ', len(use_embed_arr))
    print('len(tsdae_embed_arr): ', len(tsdae_embed_arr))
    print('len(labels): ', len(labels))
        
    use_embed_arr = {'input_1': np.array(use_embed_arr)}
    
    tsdae_embed_arr = tf.convert_to_tensor(np.array(tsdae_embed_arr), dtype=tf.float32)
    tsdae_embed_arr = {'input_2': tsdae_embed_arr}
    
    use_tsdae_embed = [use_embed_arr, tsdae_embed_arr]
    
    print('len(use_tsdae_embed): ', len(use_tsdae_embed))
    
    unique_labels = np.unique(labels)
    print('len(unique_labels): ', len(labels))
    print('unique_labels:', unique_labels)
    
    #oos_index = np.where(unique_labels == oos_intent_str)[0][0]
    #print('oos_index:', oos_index)

    return use_tsdae_embed