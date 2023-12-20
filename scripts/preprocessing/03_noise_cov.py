#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:18:56 2023

@author: rl05
"""

import os.path as op

from mne import compute_covariance, read_epochs

import config

subjects = config.subject_ids

data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')

# noise cov parameters
tmin = -0.5
tmax_cov = -0.3
method = 'auto'

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
    
# create two noise covariance matrices, one for just MEG, one for EEG & MEG
ch_types = ['MEEG', 'MEG']

for ch_type in ch_types:

    noise_cov_fname = op.join(epoch_path, f'{subject}_{ch_type}_cov.fif')
    
    if op.exists(noise_cov_fname):
        print(f'noise cov {ch_type} exists.')
        pass
    else:
        print(f'nois cov {ch_type} does not exist. Computing.')
        if epochs.preload:
            pass
        else:
            epochs.load_data()
            
        if ch_type == 'MEG':
            epochs = epochs.pick(picks='meg')
        
        noise_cov = compute_covariance(
            epochs, tmin=tmin, tmax=tmax_cov, method=method, rank='info'
            )
    
        noise_cov.save(noise_cov_fname, overwrite=True)
        
        fig_cov, fig_svd = noise_cov.plot(epochs.info, show=False)
        fig_cov.savefig(op.join(epoch_path, f'fig_cov_{ch_type}.png'))
        fig_svd.savefig(op.join(epoch_path, f'fig_svd_{ch_type}.png'))
        
        # evaluate regularisation
        fig_evoked_whitened = epochs.average().plot_white(noise_cov, time_unit="s", show=False)
        fig_evoked_whitened.savefig(op.join(epoch_path, f'fig_evoked_whitened_{ch_type}.png'))