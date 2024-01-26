# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:27:06 2023

@author: hossa
"""
from timeit import default_timer as timer

start = timer()

import numpy as np
import tensorflow_hub as hub
import keras
from tensorflow.keras.optimizers import Adam
from sentence_transformers import SentenceTransformer
from sklearn.utils import class_weight

#from load_dataset import prepare_clinc_150_dataset
from load_dataset_fixed_intents import prepare_clinc_150_known_ratio_fixed_intents, prepare_known_ratio_fixed_intents_labeled_ratio
from utils_functions import construction_outliers, prepare_targets, get_lr_metric, EarlyStoppingValAcc
from create_model import model_create
from data_embeddings import train_data_embeddings, evaluate_data_embeddings
from training import compute_threshold_dev, model_testing, model_testing_in_oos, model_testing_threshold, model_testing_threshold_in_oos

import tensorflow as tf
import configs

from utils_functions import read_tsv_file

THRESHOLDS = [i * 0.02 for i in range(0, 50)]

percentage_max_accuracy_arr = [-1, -1, -1, 1, 0.95, 0.9, 0.85]
labeled_ratio_arr = [0.2, 0.4, 0.6, 0.8, 1] # [0.2, 0.4, 0.6, 0.8, 1]
known_cls_ratio_arr = [0.75] #[0.25, 0.50, 0.75]
open_domain_train_size_arr = [1] #[1, 10, 100] #[1, 5, 10, 100, 500, 50, 100, 250, 100, 400, 500, 1000, 1000, 4000, 2000]
synthetic_outliers_train_size_arr = [1] #[1, 10, 100] #[1, 5, 10, 100, 500, 50, 100, 250, 400, 100, 500, 1000, 4000, 1000, 8000]

seed_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

date = 'stackoverflow_with_unselected_intents_train_test_as_oos_20_01_2024_001'

options = configs.parse_args()

batch_sz = options.batch_sz
epochs_num = options.epochs_num
patience_thr = options.patience_thr

oos_intent_str = options.oos_intent_str
in_domain_intent_str = options.in_domain_intent_str
num_runs = options.num_runs

opt = Adam()
lr_track = get_lr_metric(opt)

oos_intent_str = 'oos'
in_domain_intent_str = 'in-domain'

#with tf.device("cpu:0"):
### Apply Universal Sentence Encoder (USE)
module_url = 'universal-sentence-encoder-large_4'#'https://tfhub.dev/google/universal-sentence-encoder-large/4'
# Import the Universal Sentence Encoder's TF Hub module
use_embed = hub.load(module_url)

### Apply Tranformer-based Denoising AutoEncoder (TSDAE)
tsdae_model_name = 'roberta_clinc150_tsdae_model_all' #'TSDAE_scidocs_model' 'roberta-base_pooling_model' 'tsdae_askubuntu_model'
tsdae_embed = SentenceTransformer(tsdae_model_name)

### Load datasets
all_train_text, all_train_labels = read_tsv_file('datasets/stackoverflow/train.tsv')
print('len(all_train_text): ', len(all_train_text))
print('len(all_train_labels): ', len(all_train_labels))

all_dev_text, all_dev_labels = read_tsv_file('datasets/stackoverflow/dev.tsv')
print('len(all_dev_text): ', len(all_dev_text))
print('len(all_dev_labels): ', len(all_dev_labels))

all_test_text, all_test_labels = read_tsv_file('datasets/stackoverflow/test.tsv')
print('len(all_test_text): ', len(all_test_text))
print('len(all_test_labels): ', len(all_test_labels))

all_neg_text, all_neg_labels = read_tsv_file('datasets/squad/squad.tsv')
print('len(all_neg_text): ', len(all_neg_text))
print('len(all_neg_labels): ', len(all_neg_labels))
    
with tf.device("cpu:0"):    
    for ii in range(len(known_cls_ratio_arr)):
        known_cls_ratio = known_cls_ratio_arr[ii]
       
        for jj in range(len(open_domain_train_size_arr)):
            open_domain_train_size = open_domain_train_size_arr[jj]
            synthetic_outliers_train_size = synthetic_outliers_train_size_arr[jj]
            
            for ll in range(len(labeled_ratio_arr)):
                labeled_ratio = labeled_ratio_arr[ll]
                
                file_name_test = f"Epochs_{epochs_num}_batch_{batch_sz}_intents_{known_cls_ratio}_open_domain_{open_domain_train_size}_synthetic_outliers_train_size_{synthetic_outliers_train_size}_patience_{patience_thr}_labeled_ratio_{labeled_ratio}_{date}.txt"
                save_model_name = f"Epochs_{epochs_num}_batch_{batch_sz}_intents_{known_cls_ratio}_open_domain_{open_domain_train_size}_synthetic_outliers_train_size_{synthetic_outliers_train_size}_patience_{patience_thr}_labeled_ratio_{labeled_ratio}_{date}"

                print('file_name_test: ', file_name_test)

                f_test = open(file_name_test, 'w')

                print('save_model_name: ', save_model_name)
                f_test.write("{}\n".format(save_model_name))

                all_f1_known_model = []
                all_acc_known_model = []
                all_f1_unknown_model = []
                all_acc_unknown_model = []

                all_f1_known_thr = []
                all_acc_known_thr = []
                all_f1_unknown_thr = []
                all_acc_unknown_thr = []

                dev_count = 0

                for kk in range(num_runs):
                    seed = seed_arr[kk]

                    ### Load dataset ###              
                    #train_text, train_labels, dev_texts, dev_labels, test_texts, test_labels, indomain_test_labels, indomain_test_texts, oos_test_labels, oos_test_texts, known_label_list = prepare_clinc_150_known_ratio_fixed_intents(f_test, known_cls_ratio, kk, all_train_text, all_train_labels, all_dev_text, all_dev_labels, all_test_text, all_test_labels, oos_intent_str, seed)
                    train_text, train_labels, dev_texts, dev_labels, test_texts, test_labels, indomain_test_labels, indomain_test_texts, oos_test_labels, oos_test_texts, known_label_list = prepare_known_ratio_fixed_intents_labeled_ratio(f_test, known_cls_ratio, kk, all_train_text, all_train_labels, all_dev_text, all_dev_labels, all_test_text, all_test_labels, oos_intent_str, seed, labeled_ratio)

                    use_embed_train_oos, tsdae_embed_train_oos, oos_train_labels = construction_outliers(train_labels, train_text, synthetic_outliers_train_size, known_label_list, all_neg_text, open_domain_train_size, use_embed, tsdae_embed, oos_intent_str)
                    
                    ### Prepare the embeddings for the model ###
                    print(' ------------------ Train data_embedding ------------------- ')
                    use_tsdae_embed_train, train_labels_all, oos_index, number_of_classes = train_data_embeddings(train_text, train_labels, use_embed_train_oos, tsdae_embed_train_oos, oos_train_labels, use_embed, tsdae_embed, oos_intent_str)

                    print(' ------------------ Dev data_embedding ------------------- ')
                    use_tsdae_embed_dev = evaluate_data_embeddings(dev_texts, dev_labels, use_embed, tsdae_embed, oos_intent_str)

                    print(' ------------------ Test data_embedding ------------------- ')
                    use_tsdae_embed_test = evaluate_data_embeddings(test_texts, test_labels, use_embed, tsdae_embed, oos_intent_str)

                    print(' ------------------ Test_indomain data_embedding ------------------- ')
                    use_tsdae_embed_test_indomain = evaluate_data_embeddings(indomain_test_texts, indomain_test_labels, use_embed, tsdae_embed, oos_intent_str)
                    
                    print(' ------------------ Test_oos data_embedding ------------------- ')
                    use_tsdae_embed_test_oos = evaluate_data_embeddings(oos_test_texts, oos_test_labels, use_embed, tsdae_embed, oos_intent_str)
                    
                    ### Prepare the labels for the model ###
                    #replace all elements equal to 'oos\n' with a new value of 'oos'
                    #all_train_labels[all_train_labels == 'oos\n'] = oos_intent_str
                    #all_dev_labels[all_dev_labels == 'oos\n'] = oos_intent_str
                    #all_train_labels[all_train_labels == 'oos\n'] = oos_intent_str

                    y_train_enc, y_dev_enc, unique_y_train = prepare_targets(train_labels, dev_labels)
                    
                    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
                    print('class_weights: ', class_weights)
                    class_weights_dict = dict(enumerate(class_weights))
                    print('class_weights_dict: ', class_weights_dict)
                    #f_test.write("{}\n".format(class_weights_dict))

                    unique_train_labels = np.unique(train_labels)
                    unique_dev_labels = np.unique(dev_labels)
                    unique_test_labels = np.unique(test_labels)
                    
                    unique_class_weights = np.unique(class_weights)
                    print('len(class_weights):', len(class_weights))

                    print('len(train_labels_all):', len(train_labels_all))
                    
                    print('len(train_labels):', len(train_labels))
                    print('len(y_train_enc):', len(y_train_enc))
                    print('len(y_train_enc[0]):', len(y_train_enc[0]))

                    print('len(unique_train_labels):', len(unique_train_labels))
                    print('len(unique_dev_labels):', len(unique_dev_labels))
                    print('len(unique_test_labels):', len(unique_test_labels))

                    print('unique_train_labels:', unique_train_labels)
                    print('unique_dev_labels:', unique_dev_labels)
                    print('unique_test_labels:', unique_test_labels)

                    ### Model creation ### 
                    model = model_create(number_of_classes)

                    ### Model training ### 
                    opt = Adam()
                    lr_track = get_lr_metric(opt)

                    callbacks = [EarlyStoppingValAcc(monitor='val_accuracy', patience = patience_thr, file=f_test)]

                    all_f1_known_model.append([])
                    all_acc_known_model.append([])
                    all_f1_unknown_model.append([])
                    all_acc_unknown_model.append([])

                    all_f1_known_thr.append([])
                    all_acc_known_thr.append([])
                    all_f1_unknown_thr.append([])
                    all_acc_unknown_thr.append([])

                    with tf.device("gpu:0"):
                        print('\n ------------- Model No. (Run No.) : ', kk+1, '------------- ')
                        f_test.write('\n---------------- Model No. (Run No.) : {} ----------------  \n'.format(kk+1))

                        model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics=['accuracy', lr_track, keras.metrics.Precision(), keras.metrics.Recall()])

                        history = model.fit(use_tsdae_embed_train, y_train_enc, validation_data=(use_tsdae_embed_dev, y_dev_enc), epochs=epochs_num, batch_size=batch_sz, callbacks=callbacks)#, class_weight=class_weights_dict)

                    if(dev_count == 0): 
                        print('\n--------------------------- Dev Results ---------------------------\n')

                        best_thr1_arr = compute_threshold_dev(model, use_tsdae_embed_dev, dev_labels, unique_y_train, oos_intent_str, THRESHOLDS, f_test, percentage_max_accuracy_arr)
                        dev_count = dev_count + 1

                    for j in range(len(best_thr1_arr)):
                        print(' \n --------- Best_thr No. ', j+1)
                        f_test.write('\n --------- Best_thr No. {} ---------\n'.format(j+1))

                        best_thr = best_thr1_arr[j]

                        print('\n ---------------- best_thr1: ', best_thr, ' ---------------- \n')
                        f_test.write('\n ---------------- best_thr1: {} ---------------- \n'.format(best_thr))

                        print('\n---------------------- Test Results - All labels using the model only ---------------------\n')
                        f_test.write('\n-------- Test Results - All labels using the model only --------\n')

                        print('\n ------------------ Test_All results ------------------- \n')
                        f1_known, f1_unknown = model_testing(model, use_tsdae_embed_test, test_labels, unique_y_train, oos_index, f_test)

                        print('\n ------------------ Test_indomain results ------------------- \n')
                        acc_indomain = model_testing_in_oos(model, use_tsdae_embed_test_indomain, indomain_test_labels, unique_y_train, f_test)

                        print('\n ------------------ Test_oos results ------------------- \n')
                        acc_oos = model_testing_in_oos(model, use_tsdae_embed_test_oos, oos_test_labels, unique_y_train, f_test)

                        print('\n ------------------------------------------- \n')
                        if f_test is not None:
                            f_test.write('acc_known: {}\n'.format(acc_indomain))
                            f_test.write('f1_known: {}\n'.format(f1_known))
                            f_test.write('acc_unknown: {}\n'.format(acc_oos))
                            f_test.write('f1_unknown: {}\n'.format(f1_unknown))

                        all_acc_known_model[kk].append(acc_indomain)        
                        all_f1_known_model[kk].append(f1_known)
                        all_acc_unknown_model[kk].append(acc_oos)
                        all_f1_unknown_model[kk].append(f1_unknown)

                        print('\n---------------------- Test Results - All labels using the model + Threshold --------------------- \n')
                        f_test.write('\n-------- Test Results - All labels using the model + Threshold -------- \n')

                        print('\n ------------------ Test_All results ------------------- \n')
                        f1_known, f1_unknown = model_testing_threshold(model, use_tsdae_embed_test, test_labels, best_thr, unique_y_train, oos_intent_str, oos_index, f_test)

                        print('\n ------------------ Test_indomain results ------------------- \n')
                        acc_indomain = model_testing_threshold_in_oos(model, use_tsdae_embed_test_indomain, indomain_test_labels, best_thr, unique_y_train, oos_intent_str, f_test)

                        print('\n ------------------ Test_oos results ------------------- \n')
                        acc_oos = model_testing_threshold_in_oos(model, use_tsdae_embed_test_oos, oos_test_labels, best_thr, unique_y_train, oos_intent_str, f_test)

                        print('\n ------------------------------------------- \n')
                        if f_test is not None:
                            f_test.write('\nacc_known: {}\n'.format(acc_indomain))
                            f_test.write('f1_known: {}\n'.format(f1_known))
                            f_test.write('acc_unknown: {}\n'.format(acc_oos))
                            f_test.write('f1_unknown: {}\n'.format(f1_unknown))

                        all_acc_known_thr[kk].append(acc_indomain)        
                        all_f1_known_thr[kk].append(f1_known)
                        all_acc_unknown_thr[kk].append(acc_oos)
                        all_f1_unknown_thr[kk].append(f1_unknown)

                all_f1_known_model = np.array(all_f1_known_model)
                all_acc_known_model = np.array(all_acc_known_model)
                all_f1_unknown_model = np.array(all_f1_unknown_model)
                all_acc_unknown_model = np.array(all_acc_unknown_model)

                all_f1_known_thr = np.array(all_f1_known_thr)
                all_acc_known_thr = np.array(all_acc_known_thr)
                all_f1_unknown_thr = np.array(all_f1_unknown_thr)
                all_acc_unknown_thr = np.array(all_acc_unknown_thr)

                print('all_f1_known_model: ', all_f1_known_model)
                print('all_acc_known_model: ', all_acc_known_model)
                print('all_f1_unknown_model: ', all_f1_unknown_model)
                print('all_acc_unknown_model: ', all_acc_unknown_model)

                print('all_f1_known_thr: ', all_f1_known_thr)
                print('all_acc_known_thr: ', all_acc_known_thr)
                print('all_f1_unknown_thr: ', all_f1_unknown_thr)
                print('all_acc_unknown_thr: ', all_acc_unknown_thr)

                print('\n --------- Final Results !!! --------- ')
                f_test.write('\n --------- Final Results !!! --------- ')

                for i in range(len(best_thr1_arr)):
                    print(' \n --------- Best_thr No. ', i+1)
                    f_test.write('\n --------- Best_thr No. {} ---------\n'.format(i+1))

                    best_thr1 = best_thr1_arr[i]
                    print('best_thr1: ', best_thr1)
                    f_test.write('\n --------- best_thr = {} ---------\n'.format(best_thr1))

                    f1_known_model = all_f1_known_model[:,i]
                    acc_known_model = all_acc_known_model[:,i]
                    f1_unknown_model = all_f1_unknown_model[:,i]
                    acc_unknown_model = all_acc_unknown_model[:,i]

                    f1_known_thr = all_f1_known_thr[:,i]
                    acc_known_thr = all_acc_known_thr[:,i]
                    f1_unknown_thr = all_f1_unknown_thr[:,i]
                    acc_unknown_thr = all_acc_unknown_thr[:,i]

                    print('np.mean(f1_known_model): ', np.mean(f1_known_model))
                    print('np.mean(acc_known_model): ', np.mean(acc_known_model))
                    print('np.mean(f1_unknown_model): ', np.mean(f1_unknown_model))
                    print('np.mean(acc_unknown_model): ', np.mean(acc_unknown_model))

                    f_test.write('\n----------------------------------------------------------------\n')
                    f_test.write('np.mean(f1_known_model): {}\n'.format(np.mean(f1_known_model)))
                    f_test.write('np.mean(acc_known_model): {}\n'.format(np.mean(acc_known_model)))
                    f_test.write('np.mean(f1_unknown_model): {}\n'.format(np.mean(f1_unknown_model)))
                    f_test.write('np.mean(acc_unknown_model): {}\n'.format(np.mean(acc_unknown_model)))

                    print('np.std(f1_known_model): ', np.std(f1_known_model))
                    print('np.std(acc_known_model): ', np.std(acc_known_model))
                    print('np.std(f1_unknown_model): ', np.std(f1_unknown_model))
                    print('np.std(acc_unknown_model): ', np.std(acc_unknown_model))

                    f_test.write('np.std(f1_known_model): {}\n'.format(np.std(f1_known_model)))
                    f_test.write('np.std(acc_known_model): {}\n'.format(np.std(acc_known_model)))
                    f_test.write('np.std(f1_unknown_model): {}\n'.format(np.std(f1_unknown_model)))
                    f_test.write('np.std(acc_unknown_model): {}\n'.format(np.std(acc_unknown_model)))

                    print('-------------------------------------')

                    print('np.mean(f1_known_thr): ', np.mean(f1_known_thr))
                    print('np.mean(acc_known_thr): ', np.mean(acc_known_thr))
                    print('np.mean(f1_unknown_thr): ', np.mean(f1_unknown_thr))
                    print('np.mean(acc_unknown_thr): ', np.mean(acc_unknown_thr))

                    f_test.write('\n----------------------------------------------------------------\n')
                    f_test.write('np.mean(f1_known_thr): {}\n'.format(np.mean(f1_known_thr)))
                    f_test.write('np.mean(acc_known_thr): {}\n'.format(np.mean(acc_known_thr)))
                    f_test.write('np.mean(f1_unknown_thr): {}\n'.format(np.mean(f1_unknown_thr)))
                    f_test.write('np.mean(acc_unknown_thr): {}\n'.format(np.mean(acc_unknown_thr)))

                    print('np.std(f1_known_thr): ', np.std(f1_known_thr))
                    print('np.std(acc_known_thr): ', np.std(acc_known_thr))    
                    print('np.std(f1_unknown_thr): ', np.std(f1_unknown_thr))
                    print('np.std(acc_unknown_thr): ', np.std(acc_unknown_thr))

                    f_test.write('np.std(f1_known_thr): {}\n'.format(np.std(f1_known_thr)))
                    f_test.write('np.std(acc_known_thr): {}\n'.format(np.std(acc_known_thr)))
                    f_test.write('np.std(f1_unknown_thr): {}\n'.format(np.std(f1_unknown_thr)))
                    f_test.write('np.std(acc_unknown_thr): {}\n'.format(np.std(acc_unknown_thr)))
            
            if f_test is not None:
                f_test.close()
                
end = timer()
total_time = (end - start) / 60
print('Elapsed Time: ', total_time, ' min')

model.save(save_model_name)