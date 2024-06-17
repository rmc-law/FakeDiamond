#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:03:53 2023

@author: rl05
"""

import os
import os.path as op

from mne import read_epochs, compute_source_morph, read_source_spaces
from mne.minimum_norm import read_inverse_operator, apply_inverse

import config 

subjects = [f'sub-{subject}' for subject in config.subject_ids]
data_dir = op.join(config.project_repo, 'data')

preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

print(f'subjects (n={len(subjects)}): ', subjects)

# for processing specific subject
subject = input('subject to process: ')

        
print(subject)
print('======')

fsaverage_src_fname = op.join(data_dir, 'mri', 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src_to = read_source_spaces(fsaverage_src_fname) 

stc_path = op.join(data_dir, 'stcs', subject)
if op.exists(stc_path):
    print(f'{subject} stc path exists.')
else:
    print(f'Making {subject} stc path.')
    os.makedirs(stc_path, exist_ok=True)

for ch_type in ['MEEG', 'MEG']:
    
    epoch_path = op.join(preprocessed_data_path, subject, 'epoch')
    epoch_fname = op.join(epoch_path, f'{subject}_epo.fif')
    inv_fname = op.join(epoch_path, f'{subject}_{ch_type}_inv.fif')
    trans_fname = op.join(subjects_dir, subject, f'{subject}_trans.fif')
    morph_fname = op.join(subjects_dir, subject, f'{subject}-morph.h5')
    
    # meg-mri coregistration
    if not op.exists(trans_fname):
        print(f'{subject} trans.fif does not exist. use `mne coreg` in command line using `module load mnelib`')
        
        
    # epochs 
    epochs = read_epochs(epoch_fname, verbose=False)
    if ch_type == 'MEG':
        epochs = epochs.pick(picks='meg')
    
    # inverse operator
    inv = read_inverse_operator(inv_fname, verbose=False)
    
    # make stcs
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    method = 'MNE'
    
    conditions = list(epochs.event_id.keys())
    for condition in conditions:
        evoked = epochs[condition].average()
        stc = apply_inverse(evoked, inv, lambda2=lambda2, method=method,
                            return_residual=False, verbose=True)
        morph = compute_source_morph(
            src=stc,
            subject_from=subject,
            subject_to='fsaverage_src', 
            src_to=src_to
            )        
        stc_morph = morph.apply(stc)
        del stc
        condition = '-'.join(condition.replace('/','-').split('-')[:-1]) # a bit of formatting for saving fname
        stc_morph.save(op.join(stc_path, f'{subject}_{condition}_{ch_type}'), 
                       overwrite=True)
        del stc_morph
        
    
    # save residual as quality assurance of source estimation
    evoked_all = epochs.average()
    _, res_all = apply_inverse(evoked_all, inv, lambda2=lambda2, method=method, return_residual=True, verbose=True)
    fig_residual = res_all.plot(show=False)
    fig_residual.savefig(op.join(stc_path, f'fig_{subject}_{ch_type}_residual_all_conditions-ave.png'))

    del evoked_all, res_all, fig_residual



#%% 

from mne import concatenate_epochs
epochs_single_word = concatenate_epochs([epochs['baseline'],
                                              epochs['low/word1'],
                                              epochs['high/word1']])
evoked_single_word = epochs_single_word.average()
stc_single_word = apply_inverse(evoked_single_word, inv,lambda2=lambda2, method=method)
stc_single_word.plot(hemi='lh', smoothing_steps=10, size=(1000, 500), time_viewer=True)

epochs_compose = concatenate_epochs([epochs['subsective'],
                                          epochs['privative'],
                                          epochs['mid']])
stc_compose = apply_inverse(epochs_compose.average(), inv, lambda2=lambda2, method=method)
stc_compose.plot(hemi='lh', smoothing_steps=10, size=(1000, 500), time_viewer=True)

t = stc_compose.data - stc_single_word.data
stc_dummy = stc_compose.copy()
stc_dummy.data = t
stc_dummy.plot(hemi='both', smoothing_steps=10, size=(1000, 500), time_viewer=True)
