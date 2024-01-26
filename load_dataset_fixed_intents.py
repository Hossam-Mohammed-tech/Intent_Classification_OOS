# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:41:04 2023

@author: hossa
"""
import numpy as np
from utils_functions import read_tsv_file

import random
import torch

oos_intent_str = 'oos'
in_domain_intent_str = 'in-domain'

def prepare_clinc_150_known_ratio_fixed_intents(f_test, known_cls_ratio, run_num, train_text, train_labels, dev_texts, dev_labels, test_texts, test_labels, oos_intent_str, seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    unique_all_train_labels = np.unique(train_labels)
    print('unique_all_train_labels:', unique_all_train_labels)
    
    number_of_classes = len(unique_all_train_labels)
    print('number_of_classes:', number_of_classes)

    number_known_classes = round(number_of_classes * known_cls_ratio)        
    print('number_known_classes: ', number_known_classes)
    
    known_label_list = np.random.choice(unique_all_train_labels, size=number_known_classes, replace=False)    
    #known_label_list = get_known_label_list(known_cls_ratio, run_num)
    
    print('len(known_label_list):', len(known_label_list))
    
    if oos_intent_str in known_label_list:
        print('OOS already exists!')
    else:
        print('OOS will be added to known_label_list!')
        known_label_list = np.append(known_label_list, oos_intent_str)

    print('\nknown_label_list:', known_label_list)
    print('len(known_label_list):', len(known_label_list))
    
    f_test.write('\nknown_label_list: {}\n'.format(known_label_list))
    f_test.write('\nlen(known_label_list): {}\n'.format(len(known_label_list)))
    
    modified_train_texts = []
    modified_train_labels = []
                
    for i in range(len(train_labels)):
        if(train_labels[i] in known_label_list):# and (np.random.uniform(0, 1) <= labeled_ratio):
            modified_train_labels.append(train_labels[i])
            modified_train_texts.append(train_text[i])
        else:
            #if(train_labels[i] == 'oos\n'):
            modified_train_labels.append(oos_intent_str)
            modified_train_texts.append(train_text[i])
    
    print('len(modified_train_texts): ', len(modified_train_texts))
    print('len(modified_train_labels): ', len(modified_train_labels))
    print('len(unique_modified_train_labels):', len(np.unique(modified_train_labels)))
    
    modified_dev_texts = []
    modified_dev_labels = []
    
    unique_dev_labels = np.unique(dev_labels)
    
    for i in range(len(dev_labels)):
        if (dev_labels[i] in known_label_list):
            modified_dev_labels.append(dev_labels[i])
            modified_dev_texts.append(dev_texts[i])
        else:
            #if(dev_labels[i] == 'oos\n'):
            modified_dev_labels.append(oos_intent_str)
            modified_dev_texts.append(dev_texts[i])
                
    print('len(modified_dev_labels): ', len(modified_dev_labels))
    print('len(unique_dev_labels):', len(np.unique(dev_labels)))
    print('unique_dev_labels:', unique_dev_labels)
             
    indomain_test_labels = []
    indomain_test_texts = []

    oos_test_labels = []
    oos_test_texts = []

    modified_test_labels = []
    modified_test_texts = []
    
    ### oos testing data + unused classes
    for i in range(len(test_labels)): #(200):
        if(test_labels[i] in known_label_list):
            indomain_test_labels.append(test_labels[i])
            indomain_test_texts.append(test_texts[i])
            modified_test_labels.append(test_labels[i])
            modified_test_texts.append(test_texts[i])
        else:
            #if(test_labels[i] == 'oos\n'):
            oos_test_labels.append(oos_intent_str)
            oos_test_texts.append(test_texts[i])
            modified_test_labels.append(oos_intent_str)
            modified_test_texts.append(test_texts[i])
        
    unique_modified_test_labels = np.unique(modified_test_labels)

    #print('unique_modified_test_labels:', unique_modified_test_labels)
    print('len(unique_modified_test_labels):', len(unique_modified_test_labels))
    print('len(train): ', len(train_labels))
    print('len(modified_train_labels): ', len(modified_train_labels))
    print('len(dev_labels): ', len(dev_labels))
    print('len(modified_dev_labels): ', len(modified_dev_labels))
    print('len(test_labels): ', len(test_labels))
    print('len(modified_test_labels): ', len(modified_test_labels))
    print('len(indomain_test_labels): ', len(indomain_test_labels))
    print('len(oos_test_labels): ', len(oos_test_labels))
    
    return modified_train_texts, modified_train_labels, modified_dev_texts, modified_dev_labels, modified_test_texts, modified_test_labels, indomain_test_labels, indomain_test_texts, oos_test_labels, oos_test_texts, known_label_list


def prepare_known_ratio_fixed_intents_labeled_ratio(f_test, known_cls_ratio, run_num, train_text, train_labels, dev_texts, dev_labels, test_texts, test_labels, oos_intent_str, seed, labeled_ratio):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    unique_all_train_labels = np.unique(train_labels)
    #print('unique_all_train_labels:', unique_all_train_labels)
    
    number_of_classes = len(unique_all_train_labels)
    #print('number_of_classes:', number_of_classes)

    number_known_classes = round(number_of_classes * known_cls_ratio)        
    #print('number_known_classes: ', number_known_classes)
    
    known_label_list = np.random.choice(unique_all_train_labels, size=number_known_classes, replace=False)    
    #known_label_list = get_known_label_list(known_cls_ratio, run_num)
    
    #print('len(known_label_list):', len(known_label_list))
    
    if oos_intent_str in known_label_list:
        print('OOS already exists!')
    else:
        print('OOS will be added to known_label_list!')
        known_label_list = np.append(known_label_list, oos_intent_str)

    print('\nknown_label_list:', known_label_list)
    print('len(known_label_list):', len(known_label_list))
    
    f_test.write('\nknown_label_list: {}\n'.format(known_label_list))
    f_test.write('\nlen(known_label_list): {}\n'.format(len(known_label_list)))
    
    modified_train_texts = []
    modified_train_labels = []
    
    train_indomain_counter = 0
    train_oos_counter = 0
    
    for i in range(len(train_labels)):
        if(train_labels[i] in known_label_list):
            if (np.random.uniform(0, 1) <= labeled_ratio):
                modified_train_labels.append(train_labels[i])
                modified_train_texts.append(train_text[i])
                train_indomain_counter = train_indomain_counter + 1
        else:
            #if(train_labels[i] == 'oos\n'):
            modified_train_labels.append(oos_intent_str)
            modified_train_texts.append(train_text[i])
            train_oos_counter = train_oos_counter + 1
    
    print('train_indomain_counter: ', train_indomain_counter)
    print('train_oos_counter: ', train_oos_counter)
    
    print('len(modified_train_texts): ', len(modified_train_texts))
    print('len(modified_train_labels): ', len(modified_train_labels))
    print('len(unique_modified_train_labels):', len(np.unique(modified_train_labels)))
    
    modified_dev_texts = []
    modified_dev_labels = []
    
    dev_indomain_counter = 0
    dev_oos_counter = 0
    
    unique_dev_labels = np.unique(dev_labels)
    
    for i in range(len(dev_labels)):
        if (dev_labels[i] in known_label_list):
            modified_dev_labels.append(dev_labels[i])
            modified_dev_texts.append(dev_texts[i])
            dev_indomain_counter = dev_indomain_counter + 1
        else:
            #if(dev_labels[i] == 'oos\n'):
            modified_dev_labels.append(oos_intent_str)
            modified_dev_texts.append(dev_texts[i])
            dev_oos_counter = dev_oos_counter + 1
    
    print('dev_indomain_counter: ', dev_indomain_counter)
    print('dev_oos_counter: ', dev_oos_counter)
    
    print('len(modified_dev_labels): ', len(modified_dev_labels))
    print('len(unique_dev_labels):', len(np.unique(dev_labels)))
    #print('unique_dev_labels:', unique_dev_labels)
            
    indomain_test_labels = []
    indomain_test_texts = []

    oos_test_labels = []
    oos_test_texts = []

    modified_test_labels = []
    modified_test_texts = []
    
    test_indomain_counter = 0
    test_oos_counter = 0
    
    ### oos testing data + unused classes
    for i in range(len(test_labels)):
        if(test_labels[i] in known_label_list):
            indomain_test_labels.append(test_labels[i])
            indomain_test_texts.append(test_texts[i])
            modified_test_labels.append(test_labels[i])
            modified_test_texts.append(test_texts[i])
            test_indomain_counter = test_indomain_counter +1
            
        else:
            #if(test_labels[i] == 'oos\n'):
            oos_test_labels.append(oos_intent_str)
            oos_test_texts.append(test_texts[i])
            modified_test_labels.append(oos_intent_str)
            modified_test_texts.append(test_texts[i])    
            test_oos_counter = test_oos_counter + 1

    print('test_indomain_counter: ', test_indomain_counter)
    print('test_oos_counter: ', test_oos_counter)
        
    unique_modified_test_labels = np.unique(modified_test_labels)

    #print('unique_modified_test_labels:', unique_modified_test_labels)
    print('len(unique_modified_test_labels):', len(unique_modified_test_labels))
    print('len(train): ', len(train_labels))
    print('len(modified_train_labels): ', len(modified_train_labels))
    print('len(dev_labels): ', len(dev_labels))
    print('len(modified_dev_labels): ', len(modified_dev_labels))
    print('len(test_labels): ', len(test_labels))
    print('len(modified_test_labels): ', len(modified_test_labels))
    print('len(indomain_test_labels): ', len(indomain_test_labels))
    print('len(oos_test_labels): ', len(oos_test_labels))
    
    return modified_train_texts, modified_train_labels, modified_dev_texts, modified_dev_labels, modified_test_texts, modified_test_labels, indomain_test_labels, indomain_test_texts, oos_test_labels, oos_test_texts, known_label_list