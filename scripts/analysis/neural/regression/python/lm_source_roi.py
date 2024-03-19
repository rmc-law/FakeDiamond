#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:44:33 2024

@author: rl05
"""


import sys
import os
import os.path as op
from itertools import compress

import numpy as np
import pandas as pd
from scipy.stats import zscore

from mne import read_epochs
from mne.stats import linear_regression

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/lm')
from lm import *

subject_ids = config.subject_ids
print(f'subjects (n={len(subject_ids)}): \n', subject_ids)

data_dir = op.join(config.project_repo, 'data')
# results_dir = op.join(config.project_repo, 'results')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
analysis_dir = op.join(config.project_repo, 'scripts/analysis/neural/lm/')
output_dir = op.join(analysis_dir, 'betas')


analysis_id = input('analysis id: ')
analysis = analysis_mapping[analysis_id][0]
experiment = analysis_mapping[analysis_id][1]
print('analysis: ', analysis)

rois = ['anteriortemporal-lh','posteriortemporal-lh',
       'inferiorfrontal-lh','temporoparietal-lh','lateraloccipital-lh']
for subject_id in subject_ids:

    subject = f'sub-{subject_id}'
    print(subject, roi)
    
    subject_output_dir = op.join(output_dir, analysis, 'source', subject)
    if op.exists(subject_output_dir):
        print(f'{subject} done. skipping.')
    else:
        os.makedirs(subject_output_dir, exist_ok=True)
        
        for roi in rois:
            # read in source estimates
            stcs_epochs_path = op.join(data_dir, 'stcs_epochs', subject)
            stcs_epochs_fname = op.join(stcs_epochs_path, f'stcs_epochs_{roi}_100Hz.stc.npy')
            if not op.exists(stcs_epochs_fname):
                pass
                print(subject, ' stcs in ', roi, ' not found.')
            else:
                stcs = list(np.load(stcs_epochs_fname, allow_pickle=True)) # stc of a given roi
            
            # get trial info
            epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
            epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')
            epochs = read_epochs(epoch_fname, preload=False, verbose=False)
            log_fname = op.join(config.logs_dir, f'{subject}_logfile.csv')
            trial_info = pd.read_csv(log_fname)
            trial_info['length_adj'] = trial_info['word1'].apply(count_letters)
            trial_info['length_noun'] = trial_info['word2'].apply(count_letters)
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
            
            # get lexical frequency and transition probability values
            stimuli = pd.read_csv(op.join(analysis_dir, 'stimuli_test.csv'))
            # trial_info['subject'] = int(subject_id)
            stimuli.index = stimuli.index + 1
            trial_info.set_index('item_nr', inplace=True)
            trial_info['zipf_adj'] = trial_info.index.map(stimuli['zipf_word1'])
            trial_info['zipf_noun'] = trial_info.index.map(stimuli['zipf_word2'])
            trial_info['freq_dep'] = trial_info.index.map(stimuli['freq_dep'])
            trial_info['freq_seq'] = trial_info.index.map(stimuli['freq_seq'])
            trial_info.reset_index()

            # get epochs drop log as a mask, then apply it to trial info as epochs.metadata
            epochs_drop_mask = [not bool(epoch) for epoch in epochs.drop_log]
            assert(epochs_drop_mask.count(True) == len(epochs.events)) # sanity
            epochs.metadata = trial_info[epochs_drop_mask]
            
            if (experiment == 'compose') or (experiment == 'specificity'):
                # subselect epochs based on experiment
                experiment_drop_mask = (epochs.metadata.experiment == experiment).values 
                epochs = epochs[experiment_drop_mask]
                stcs = list(compress(stcs, experiment_drop_mask))
                assert len(epochs) == len(stcs)
                
            if analysis == 'denotation':
                # drop baseline conditions, to compare between only subsective vs. privative
                baseline_drop_mask = (epochs.metadata.denotation == 'baseline').values 
                baseline_drop_mask = [not t for t in baseline_drop_mask] # flip truth values
                epochs = epochs[baseline_drop_mask]
                stcs = list(compress(stcs, baseline_drop_mask))
                assert len(epochs) == len(stcs)

                
            # set up design matrix
            if experiment == 'compose':
                covariates_vars = ['zipf_adj', 'length_adj', 'freq_seq'] 
                covariates_vars = []
                predictor_zipf_adj = zscore(epochs.metadata[['zipf_adj']].values.ravel()).tolist()
                predictor_length_adj = epochs.metadata[['length_adj']].values.ravel().tolist()
                predictor_freq_seq = zscore(epochs.metadata[['freq_seq']].values.ravel()).tolist()
                predictors_covariates = [predictor_zipf_adj, predictor_length_adj, predictor_freq_seq]
                predictors_covariates = []
            elif experiment == 'specificity':
                covariates_vars = ['zipf_adj','zipf_noun','length_adj','length_noun'] 
                predictor_zipf_adj = zscore(epochs.metadata[['zipf_adj']].values.ravel()).tolist()
                predictor_zipf_noun = [0. if np.isnan(z) else z for z in zscore(epochs.metadata[['zipf_noun']].values.ravel()).tolist()]
                predictor_length_adj = epochs.metadata[['length_adj']].values.ravel().tolist()
                predictor_length_noun = epochs.metadata[['length_noun']].values.ravel().tolist()
                predictors_covariates = [predictor_zipf_adj, predictor_zipf_noun, predictor_length_adj, predictor_length_noun]
            predictor_intercept = [1] * len(epochs)

            if analysis == 'compose':
                predictor_names = covariates_vars + ['intercept','composition','concreteness','denotation',
                                                    'composition:concreteness','denotation:concreteness']#'composition:denotation']#,'composition:concreteness:denotation']
                predictor_composition = encode_labels(epochs.metadata[['composition']].values.tolist(), 'composition')   
                predictor_concreteness = encode_labels(epochs.metadata[['concreteness']].values.tolist(), 'concreteness')  
                predictor_denotation = encode_labels(epochs.metadata[['denotation']].values.tolist(), 'denotation')   
                predictor_compositionXconcreteness = list(map(lambda a, b: a * b, predictor_composition, predictor_concreteness))
                # predictor_compositionXdenotation = list(map(lambda a, b: a * b, predictor_composition, predictor_denotation))
                predictor_denotationXconcreteness = list(map(lambda a, b: a * b, predictor_concreteness, predictor_denotation))
                # predictor_interaction_full = list(map(lambda a, b, c: a * b * c, predictor_composition, predictor_concreteness, predictor_denotation))
                predictors_all = [*predictors_covariates, predictor_intercept, predictor_composition, predictor_concreteness, predictor_denotation,
                                predictor_compositionXconcreteness, predictor_denotationXconcreteness]#, predictor_compositionXdenotation]#, predictor_interaction_full]
            elif analysis == 'specificity':
                predictor_names = covariates_vars + ['intercept','composition','specificity']
                predictor_composition = encode_labels(epochs.metadata[['composition']].values.tolist(), 'composition')   
                predictor_specificity = encode_labels(epochs.metadata[['specificity']].values.tolist(), 'specificity')            
                predictors_all = [*predictors_covariates, predictor_intercept, predictor_composition, predictor_specificity]
            design = pd.DataFrame(list(zip(*predictors_all)), columns=predictor_names)

        
            # run linear model
            reg = linear_regression(stcs, design_matrix=design, names=predictor_names)
            for predictor in predictor_names: 
                if predictor != 'intercept':
                    reg[predictor].beta.save(op.join(subject_output_dir, f'betas_{roi}_{predictor}'), verbose=False)
            del reg
        
    print('\n')
