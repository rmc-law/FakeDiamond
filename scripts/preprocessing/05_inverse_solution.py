#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:26:20 2023

@author: rl05
"""


import os.path as op

from mne import read_epochs, read_forward_solution, read_cov, set_log_level
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

set_log_level(verbose=False)

import config


data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
print('subjects: ', subjects)
source_space = input('source space (ico4 or oct6): ')

for orientation in ['loose','fixed']:
    for subject in subjects:
        
        if source_space == 'ico4':
            src_fname = op.join(subjects_dir, subject, f'{subject}_ico4_src.fif')
            src_suffix = '_ico4'
        elif source_space == 'oct6':
            src_fname = op.join(subjects_dir, subject, f'{subject}_oct6_src.fif')
            src_suffix = ''

    
        epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
        epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')
    
        # check whether epochs computed for this subject
        if not op.exists(epoch_fname):
            print(f'{subject} epochs.fif does not exist.')
        else:
            epochs = read_epochs(epoch_fname, preload=False, verbose=False)
            
        # create two inverse solutions, one for just MEG, one for EEG & MEG
        ch_types = ['MEEG', 'MEG']
    
        for ch_type in ch_types:
    
            if orientation == 'fixed':
                inv_fname = op.join(epoch_path, f'{subject}_{ch_type}{src_suffix}_fixed_inv.fif')
            else:
                inv_fname = op.join(epoch_path, f'{subject}_{ch_type}{src_suffix}_inv.fif')
            fwd_fname = op.join(epoch_path, f'{subject}_{ch_type}{src_suffix}_fwd.fif')
            cov_fname = op.join(epoch_path, f'{subject}_{ch_type}_cov.fif')
    
            if op.exists(inv_fname):
                print(subject, ch_type, source_space, orientation, 'inv exists. Skipping.')
                pass
            else:
                print(subject, 'making inv: ', orientation, ch_type, source_space)
                # check whether forward solution computed for this subject
                if not op.exists(fwd_fname):
                    print(f'{subject} fwd.fif does not exist.')
                    continue
                else:
                    fwd = read_forward_solution(fwd_fname, verbose=False)
    
                # check whether noise cov computed for this subject
                if not op.exists(cov_fname):
                    print(f'{subject} cov.fif does not exist.')
                    continue
                else:
                    noise_cov = read_cov(cov_fname, verbose=False)
    
                if epochs.preload:
                    pass
                else:
                    epochs.load_data()
    
                if ch_type == 'MEG':
                    epochs = epochs.pick(picks='meg')
                
                if orientation == 'fixed':
                    loose = 0.
                    depth = None
                    fixed = True
                else:
                    loose = 0.2
                    depth = None
                    fixed = 'auto'
    
                inv = make_inverse_operator(
                    epochs.info, fwd, noise_cov,
                    loose=loose, 
                    depth=depth, 
                    fixed=fixed,
                    rank='info'
                    )
                
                write_inverse_operator(inv_fname, inv, overwrite=True, verbose=False)
    
                del inv
        