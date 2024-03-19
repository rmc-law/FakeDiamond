#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:31:26 2024

@author: rl05

Config file for linear regression analysis on sensor and source data
"""

from itertools import chain

print(
'''
Analysis to perform (enter the index):
    1. compose
    2. specificity
'''      
)
    
analysis_mapping = {
    '1': ('compose','compose'),
    '2': ('specificity','specificity'),
    }

def count_letters(text):
    return sum(c.isalpha() for c in text)    
    
def encode_labels(source_labels, analysis):
    source_labels = list(chain(*source_labels))
    if analysis == 'concreteness':
        label_mapping = dict(concrete=0, abstract=1)
    elif analysis == 'denotation':
        label_mapping = dict(baseline=0, subsective=1, privative=2)
    elif analysis == 'specificity':
        label_mapping = dict(low=0, mid=1, high=2)
    elif analysis == 'composition':
        label_mapping = dict(word=0, phrase=1)
    target_labels = list(map(label_mapping.get, source_labels))
    return target_labels
