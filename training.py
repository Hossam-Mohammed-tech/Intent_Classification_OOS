# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:48:54 2023

@author: hossam
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from utils_functions import compute_metric_indomain_oos_dev_1_thr, domain_preds_function_test_1_thr

def compute_threshold_dev(model, use_tsdae_embed_dev, dev_labels, unique_y_train, oos_intent_str, THRESHOLDS, f_test, percentage_max_accuracy_arr):
    model_output_dev = model.predict(use_tsdae_embed_dev)
    
    print(type(model_output_dev))
    print(model_output_dev)
    print(model_output_dev.shape)
    
    f_test.write('\n-------- Dev Results --------\n')
    
    best_thr1, best_score_thr1 = compute_metric_indomain_oos_dev_1_thr(dev_labels, model_output_dev, unique_y_train, oos_intent_str, THRESHOLDS, f_test, percentage_max_accuracy_arr)
    
    f_test.write('\nbest_thr1: {},\n   -  best_score_thr1: {}\n'.format(best_thr1, best_score_thr1))
    
    print('best_thr1: ', best_thr1)
    print('best_score_thr1: ', best_score_thr1)
    
    return best_thr1

###########

def model_testing(model, use_tsdae_embed, labels, unique_y_train, oos_index, f_test):
    
    model_output = model.predict(use_tsdae_embed)

    all_preds = []

    for i in range(len(model_output)):        
        model_pred = model_output[i]
        max_model_index = np.argmax(model_pred)
        pred = unique_y_train[max_model_index]
        all_preds.append(pred)
        
    acc_score = accuracy_score(labels, all_preds)
    acc_score = acc_score * 100

    precision_score, recall_score, f_score, support = precision_recall_fscore_support(labels, all_preds, average='weighted')
    precision_score = precision_score * 100
    recall_score = recall_score * 100

    f1_scores = f1_score(labels, all_preds, average=None)
    f1_scores = f1_scores * 100

    f1_scores_mean = np.mean(f1_scores)
    f1_oos = f1_scores[oos_index]
    
    scores_1 = f1_scores[:oos_index] 
    scores_2 = f1_scores[oos_index+1 :]
    scores_known = np.append(scores_1, scores_2)
    f1_known = np.mean(scores_known)

    print('f1_scores: ', f1_scores)
    print('f1_scores_mean: ', f1_scores_mean)
    print('scores_known: ', scores_known)
    
    print('acc_score: ', acc_score, ' %')
    print('precision_score: ', precision_score, ' %')
    print('recall_score: ', recall_score, ' %')
    
    print('f1_known: ', f1_known, ' %')
    print('acc_score: ', acc_score, ' %')
    print('f1_unknown: ', f1_oos, ' %')

    print('\n ------------------------------------------- \n')

    #if f_test is not None:
    #    f_test.write('\n acc_score: {}\n'.format(acc_score))
    #    f_test.write('precision_score: {}\n'.format(precision_score))
    #    f_test.write('recall_score: {}\n'.format(recall_score))
    #    f_test.write('f1_scores_mean: {}\n'.format(f1_scores_mean))
    #    f_test.write('acc_score: {}\n'.format(acc_score))
    #    f_test.write('f1_known: {}\n'.format(f1_known))
    #    f_test.write('f1_unknown: {}\n'.format(f1_oos))
        
    return f1_known, f1_oos

def model_testing_in_oos(model, use_tsdae_embed, labels, unique_y_train, f_test):
    
    model_output = model.predict(use_tsdae_embed)

    all_preds = []

    for i in range(len(model_output)):        
        model_pred = model_output[i]
        max_model_index = np.argmax(model_pred)
        pred = unique_y_train[max_model_index]
        all_preds.append(pred)
        
    acc_score = accuracy_score(labels, all_preds)
    acc_score = acc_score * 100

    precision_score, recall_score, f_score, support = precision_recall_fscore_support(labels, all_preds, average='weighted')
    precision_score = precision_score * 100
    recall_score = recall_score * 100
    
    print('acc_score: ', acc_score, ' %')
    print('precision_score: ', precision_score, ' %')
    print('recall_score: ', recall_score, ' %')
    
    print('\n ------------------------------------------- \n')

    #if f_test is not None:
    #    f_test.write('\nacc_score: {}\n'.format(acc_score))
    #    f_test.write('precision_score: {}\n'.format(precision_score))
    #    f_test.write('recall_score: {}\n'.format(recall_score))
    
    return acc_score


def model_testing_threshold(model, use_tsdae_embed, labels, best_thr, unique_y_train, oos_intent_str, oos_index, f_test):
    
    model_output = model.predict(use_tsdae_embed)

    all_preds = domain_preds_function_test_1_thr(labels, model_output, best_thr, unique_y_train, oos_intent_str, f_test)
    
    acc_score = accuracy_score(labels, all_preds)
    acc_score = acc_score * 100

    precision_score, recall_score, f_score, support = precision_recall_fscore_support(labels, all_preds, average=None)
    precision_score_macro, recall_score_macro, f_score_macro, support_macro = precision_recall_fscore_support(labels, all_preds, average='macro')
    
    f1_scores = f1_score(labels, all_preds, average=None)
    f1_scores_macro = f1_score(labels, all_preds, average='macro')
    
    recall_score = recall_score * 100
    precision_score = precision_score * 100
  
    recall_score_macro = recall_score_macro * 100
    precision_score_macro = precision_score_macro * 100
    
    f1_scores = f1_scores * 100
    f1_scores_macro = f1_scores_macro * 100
    
    f1_oos = f1_scores[oos_index]
    scores_1 = f1_scores[:oos_index]
    scores_2 = f1_scores[oos_index+1 :]
    scores_known = np.append(scores_1, scores_2)
    f1_known = np.mean(scores_known)
    
    print('acc_score: ', acc_score, ' %')
    
    print('precision_score: ', precision_score, ' %')
    print('recall_score: ', recall_score, ' %')
    
    print('precision_score_macro: ', precision_score_macro, ' %')
    print('recall_score_macro: ', recall_score_macro, ' %')
    
    print('f1_known: ', f1_known, ' %')
    print('f1_unknown: ', f1_oos, ' %')
    
    print('f1_scores_macro: ', f1_scores_macro, ' %')
    
    f_score = f_score * 100
    f_score_macro = f_score_macro * 100
    
    f1_support_oos = f_score[oos_index]
    scores_support_1 = f_score[:oos_index]
    scores_support_2 = f_score[oos_index+1 :]
    scores_support_known = np.append(scores_support_1, scores_support_2)
    f1_support_known = np.mean(scores_support_known)

    print('f1_support_known: ', f1_support_known, ' %')
    print('f1_support_unknown: ', f1_support_oos, ' %')
    
    print('f1_score_support_macro: ', f_score_macro, ' %')

    print('\n ------------------------------------------- \n')

    #if f_test is not None:
    #    f_test.write('\nacc_score: {}\n'.format(acc_score))
    #    f_test.write('precision_score: {}\n'.format(precision_score))
    #    f_test.write('recall_score: {}\n'.format(recall_score))
    #    f_test.write('f1_known: {}\n'.format(f1_known))
    #    f_test.write('f1_unknown: {}\n'.format(f1_oos))
    
    return f1_known, f1_oos

def model_testing_threshold_in_oos(model, use_tsdae_embed, labels, best_thr, unique_y_train, oos_intent_str, f_test):
    
    model_output = model.predict(use_tsdae_embed)

    all_preds = domain_preds_function_test_1_thr(labels, model_output, best_thr, unique_y_train, oos_intent_str, f_test)
    
    acc_score = accuracy_score(labels, all_preds)
    acc_score = acc_score * 100

    precision_score, recall_score, f_score, support = precision_recall_fscore_support(labels, all_preds, average='weighted')
    precision_score = precision_score * 100
    recall_score = recall_score * 100
 
    print('acc_score: ', acc_score, ' %')
    print('precision_score: ', precision_score, ' %')
    print('recall_score: ', recall_score, ' %')
    print('f1_score: ', f_score, ' %')
    
    print('\n ------------------------------------------- \n')

    #if f_test is not None:
    #    f_test.write('\nacc_score: {}\n'.format(acc_score))
    #    f_test.write('precision_score: {}\n'.format(precision_score))
    #    f_test.write('recall_score: {}\n'.format(recall_score))
    
    return acc_score