#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:29:15 2023

@author: rl05
"""


import os.path as op

from mne import (read_epochs, make_forward_solution, write_forward_solution,
                 read_source_spaces)

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
trans_fname = op.join(subjects_dir, subject, f'{subject}_trans.fif')
src_fname = op.join(subjects_dir, subject, f'{subject}_oct6_src.fif')

# check whether epochs computed for this subject
if not op.exists(epoch_fname):
    print(f'{subject} epochs.fif does not exist.')
else:
    epochs = read_epochs(epoch_fname, preload=False, verbose=False)

# check whether src computed for this subject
if not op.exists(src_fname):
    print(f'{subject} src does not exist.')
else:
    src = read_source_spaces(src_fname, verbose=False)
    

# create two forward solution, one for just MEG, one for EEG & MEG
ch_types = ['MEEG', 'MEG']

for ch_type in ch_types:

    fwd_fname = op.join(epoch_path, f'{subject}_{ch_type}_fwd.fif')
    bem_fname = op.join(subjects_dir, subject, 'bem', f'{subject}_{ch_type}-bem-sol.fif')

    if op.exists(fwd_fname):
        print(f'forward solution {ch_type} exists.')
        pass
    else:
        print(f'forward solution {ch_type} does not exist. Computing.')
        if epochs.preload:
            pass
        else:
            epochs.load_data()
            
        if ch_type == 'MEG':
            epochs = epochs.pick(picks='meg')
            meg = True
            eeg = False
        if ch_type == 'MEEG':
            meg = True
            eeg = True
        
        print ('Making forward solution.')
        fwd = make_forward_solution(
            epochs.info, trans_fname, src, bem_fname, 
            meg=meg, 
            eeg=eeg, 
            mindist=5.0, 
            verbose=True
            )
        
        print ('Writing forward solution.')
        write_forward_solution(fwd_fname, fwd, overwrite=True)