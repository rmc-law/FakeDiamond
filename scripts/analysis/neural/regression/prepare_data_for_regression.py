#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:07:13 2024

@author: rl05
"""

import os.path as op
from glob import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 

subject_ids = config.subject_ids

data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
analysis_dir = op.join(config.project_repo, 'scripts/analysis/neural/regression')
analysis_input_dir = op.join(analysis_dir, 'input')
word_two_only = input('Analyse word 2 only? (y/n): ').lower().strip() == 'y'


trial_info_group = []

for subject_id in subject_ids:

    subject = f'sub-{subject_id}'
    stcs_epochs_path = op.join(data_dir, 'stcs_epochs', subject)
    epochs_log_fname = op.join(stcs_epochs_path, 'epochs_matched_logfile.csv')

    if not op.exists(epochs_log_fname):
        pass
    else:
        stimuli = pd.read_csv(op.join(analysis_dir, 'stimuli_test.csv'))
        trial_info = pd.read_csv(epochs_log_fname)
        trial_info['subject'] = int(subject_id)
        stimuli.index = stimuli.index + 1
        trial_info.set_index('item_nr', inplace=True)
        trial_info['zipf_adj'] = trial_info.index.map(stimuli['zipf_word1'])
        trial_info['zipf_noun'] = trial_info.index.map(stimuli['zipf_word2'])
        trial_info['freq_dep'] = trial_info.index.map(stimuli['freq_dep'])
        trial_info['freq_seq'] = trial_info.index.map(stimuli['freq_seq'])

        trial_info.reset_index(inplace=True)
        # trial_info_group.append(trial_info)

        # add various contrast info
        composition_coding_dict = {'low': 'word', 'mid': 'phrase', 'high': 'word',
                                   '^(.*)_baseline$': 'word', '^(.*)_subsective$': 'phrase', '^(.*)_privative$': 'phrase'}
        trial_info['composition'] = trial_info['condition'].replace(composition_coding_dict, regex=True)
        denotation_coding_dict = {'low': '', 'mid': '', 'high': '',
                                   '^(.*)_baseline$': 'baseline', '^(.*)_subsective$': 'subsective', '^(.*)_privative$': 'privative'}
        trial_info['denotation'] = trial_info['condition'].replace(denotation_coding_dict, regex=True)
        concreteness_coding_dict = {'low': '', 'mid': '', 'high': '',
                                   '^concrete_(.*)$': 'concrete', '^abstract_(.*)$': 'abstract'}
        trial_info['concreteness'] = trial_info['condition'].replace(concreteness_coding_dict, regex=True)
        trial_info['specificity'] = np.where(trial_info['condition'].isin(['low', 'mid', 'high']), trial_info['condition'], '')
        trial_info_group.append(trial_info)
trial_info_group = pd.concat(trial_info_group, ignore_index=True)
trial_info_group.to_csv(op.join(analysis_input_dir, f'lmer_trial_info_group_(n={len(np.unique(trial_info_group["subject"]))}).csv'))

#%% 

# prepare stc array
rois = ['anteriortemporal-lh', 'posteriortemporal-lh',
        'inferiorfrontal-lh', 'temporoparietal-lh', 'lateraloccipital-lh']

for roi in rois:

    print(roi)
    data_group = []
    
    if word_two_only:
        fname_ending = '_word2only.csv'
    else:
        fname_ending = '.csv'
    
    group_data_output_fname = glob(op.join(analysis_input_dir, f'lmer_{roi}_data_group_*'))

    if group_data_output_fname != []:
        print(f'{roi} group data computed. Skipping.')
    else:
    
        for subject_id in tqdm(subject_ids, unit='subject_id'):
    
            subject = f'sub-{subject_id}'
            stcs_epochs_path = op.join(data_dir, 'stcs_epochs', subject)
            stcs_epochs_fname = op.join(stcs_epochs_path, f'stcs_epochs_{roi}.stc.npy')
            if not op.exists(stcs_epochs_fname):
                pass
            else:
                stcs_roi = np.load(stcs_epochs_fname, allow_pickle=True)
                if word_two_only:
                    stcs_roi = [stc.crop(tmin=0.6,tmax=1.4).data.mean(axis=0) for stc in stcs_roi] # average within roi
                else:
                    stcs_roi = [stc.data.mean(axis=0) for stc in stcs_roi] # average within roi
                
                num_timepoints = stcs_roi[0].shape[0]
                num_trials = len(stcs_roi)
                timepoint = np.tile(np.arange(1, num_timepoints+1), num_trials)
                trial_nr = np.repeat(np.arange(1, num_trials+1), num_timepoints)
                stcs_roi = np.array(stcs_roi).flatten() # each element is a stc time point
                subject = np.repeat(int(subject_id), len(stcs_roi))
                data = pd.DataFrame({'MNE': stcs_roi,
                                    'timepoint': timepoint,
                                    'trial_nr': trial_nr,
                                    'subject': subject})
                data_group.append(data)
            
        data_group = pd.concat(data_group)
        data_group.to_csv(op.join(analysis_input_dir, f'lmer_{roi}_data_group_(n={len(np.unique(data_group["subject"]))})'+fname_ending)) 

print('Done.')