# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:54:51 2023

@author: hossa
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'NLP experiment')
    parser.add_argument("--num_runs", default = 10, type=int, help="number of runs")
    parser.add_argument("--epochs_num", default = 1000, type=int, help="number of epochs")
    parser.add_argument("--batch_sz", default = 200, type=int, help="batch size")
    parser.add_argument("--patience_thr", default = 100, type=int, help="patience thr")
    parser.add_argument("--oos_intent_str", default = 'oos', type=str, help="oos intent str")
    parser.add_argument("--in_domain_intent_str", default = 'in-domain', type=str, help="in-domain intent str")
    
    return parser.parse_args()