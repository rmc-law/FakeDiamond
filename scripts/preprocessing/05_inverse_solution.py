#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:26:20 2023

@author: rl05
"""


import os.path as op

from mne import read_epochs, read_forward_solution, read_cov
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

import config

subjects = config.subject_ids

data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')

# for processing single subject
subject = input('subject to process: ')


subject = f'sub-{subject}'
print(subject)
print('======')

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

    inv_fname = op.join(epoch_path, f'{subject}_{ch_type}_inv.fif')
    fwd_fname = op.join(epoch_path, f'{subject}_{ch_type}_fwd.fif')
    cov_fname = op.join(epoch_path, f'{subject}_{ch_type}_cov.fif')

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


    if op.exists(inv_fname):
        print(f'inverse solution {ch_type} exists.')
        pass
    else:
        print(f'inverse solution {ch_type} does not exist. Computing.')
        if epochs.preload:
            pass
        else:
            epochs.load_data()

        if ch_type == 'MEG':
            epochs = epochs.pick(picks='meg')
        
        inv = make_inverse_operator(
            epochs.info, fwd, noise_cov,
            loose=0.2, 
            depth=None, 
            fixed='auto',
            rank='info'
            )
        
        write_inverse_operator(inv_fname, inv, overwrite=True, verbose=False)
