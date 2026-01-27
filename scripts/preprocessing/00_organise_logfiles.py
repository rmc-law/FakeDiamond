#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:58:12 2023

@author: rl05
"""

# import sys
import os.path as op
import numpy as np
import pandas as pd

# sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config


def count_letters(text):
    return sum(c.isalpha() for c in text)    
    

subject_ids = config.subject_ids
runs = [1,2,3,4,5]

# from stimulus presentation folder
logs_dir_ownCloud = '/home/rl05/ownCloud/projects/fake_diamond/scripts/stimulus_presentation/logs'

# for saving a copy to CBU server
logs_dir_cbu = config.logs_dir

# for saving a copy to ownCloud for analysis
# analysis_dir_ownCloud = '/home/rl05/ownCloud/projects/fake_diamond/scripts/analysis/behavioural/data'


# for subjects who mixed up the buttons
def unmix_buttons(logfile_df):
    logfile_df['response'] = logfile_df['response'].replace({'Ry': 'Rb', 'Rb': 'Ry'})
    return logfile_df


# get logs from each subject and put into one df
for subject_id in subject_ids:
        
    # subject_logs_fname = op.join(analysis_dir_ownCloud, f'sub-{subject_id}_logfile.csv')
    subject_logs_fname = op.join(logs_dir_cbu, f'sub-{subject_id}_logfile.csv')

    if op.exists(subject_logs_fname):
        print(f'sub-{subject_id} logfiles already organised. Skipped.')
    else: 
        # annoyingly i've created different ways of naming files so now i have to deal with it
        subject_number = config.map_subjects_meg[subject_id][0].split('_')[1]
    
        # put run-specific logfiles into one df 
        subject_logs = []
        for run in runs:            
            log_fname = op.join(logs_dir_ownCloud, subject_number, f'logfile_subject{subject_number}_block{run}.csv')
            log_tmp = pd.read_csv(log_fname, index_col=False)
            subject_logs.append(log_tmp)
            del log_tmp        
        subject_logs = pd.concat(subject_logs)
        subject_logs.insert(0, 'participant', subject_id) # add subject
        print(f'sub-{subject_id} total questions answered:', len(subject_logs) ,'out of 900')
        
    
        # add condition columns
        stimuli = pd.read_csv('/home/rl05/ownCloud/projects/fake_diamond/scripts/stimulus_presentation/stimuli_test.csv', index_col=False)
        stimuli.index = stimuli.index + 1
        subject_logs.set_index('item_nr', inplace=True)
        subject_logs['condition'] = subject_logs.index.map(stimuli['condition_name'])
        subject_logs['experiment'] = subject_logs.index.map(stimuli['experiment'])
        subject_logs['experiment'] = subject_logs['experiment'].replace('concreteness_denotation','compose')
        subject_logs['zipf_adj'] = subject_logs.index.map(stimuli['zipf_word1'])
        subject_logs['zipf_noun'] = subject_logs.index.map(stimuli['zipf_word2'])
        subject_logs['freq_dep'] = subject_logs.index.map(stimuli['freq_dep'])
        subject_logs['freq_seq'] = subject_logs.index.map(stimuli['freq_seq'])
        subject_logs.reset_index(inplace=True)


        # get word length
        subject_logs['length_adj'] = subject_logs['word1'].apply(count_letters)
        subject_logs['length_noun'] = subject_logs['word2'].apply(count_letters)
        
        # organise condition columns a bit better
        composition_coding_dict = {'low': 'word', 'mid': 'phrase', 'high': 'word',
                                    '^(.*)_baseline$': 'word', '^(.*)_subsective$': 'phrase', '^(.*)_privative$': 'phrase'}
        subject_logs['composition'] = subject_logs['condition'].replace(composition_coding_dict, regex=True)
        denotation_coding_dict = {'low': '', 'mid': '', 'high': '',
                                    '^(.*)_baseline$': 'baseline', '^(.*)_subsective$': 'subsective', '^(.*)_privative$': 'privative'}
        subject_logs['denotation'] = subject_logs['condition'].replace(denotation_coding_dict, regex=True)
        concreteness_coding_dict = {'low': '', 'mid': '', 'high': '',
                                    '^concrete_(.*)$': 'concrete', '^abstract_(.*)$': 'abstract'}
        subject_logs['concreteness'] = subject_logs['condition'].replace(concreteness_coding_dict, regex=True)
        subject_logs['specificity'] = np.where(subject_logs['condition'].isin(['low', 'mid', 'high']), subject_logs['condition'], '')
        
        # save to CBU server
        subject_logs_fname = op.join(logs_dir_cbu, f'sub-{subject_id}_logfile.csv')
        subject_logs.to_csv(subject_logs_fname, index=False)
