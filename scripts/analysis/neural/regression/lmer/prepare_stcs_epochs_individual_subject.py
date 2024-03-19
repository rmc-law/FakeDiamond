#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:04:48 2024

@author: rl05
"""

import os
import os.path as op
import sys
import numpy as np
import pandas as pd

import mne

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 

data_dir = op.join(config.project_repo, 'data')
subjects_dir = op.join(data_dir, 'mri')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
os.environ['SUBJECTS_DIR'] = subjects_dir

ch_type = 'MEEG'
resample = input('resample data? input Hz')
orientation = input('loose or fixed: ')

subjects = config.subject_ids
for subject in subjects:
    
    subject = f'sub-{subject}'
    stcs_epochs_path = op.join(data_dir, 'stcs_epochs', subject)
    
    # get subject-specific roi labels
    labels = mne.read_labels_from_annot('fsaverage_src', parc='fake_diamond', hemi='lh', verbose=False)[:-1] 
    labels_subject = mne.morph_labels(labels, subject_to=subject, subject_from='fsaverage_src', verbose=False)
    mne.write_labels_to_annot(labels_subject, subject=subject, parc='fake_diamond', verbose=False, overwrite=True)
    del labels
    assert len(labels_subject) == 5 # should be four rois + a visual sanity roi

    label_names = ['anteriortemporal-lh', 'posteriortemporal-lh',
                'inferiorfrontal-lh', 'temporoparietal-lh','lateraloccipital-lh']

    for label, label_name in zip(labels_subject, label_names):
        
        if resample:
            resample_fname_suffix = f'_{resample}Hz.stc.npy'
        else:
            resample_fname_suffix = '.stc.npy'
        
        if orientation == 'fixed':
            orientation_fname_suffix = f'_{orientation}'
        else:
            orientation_fname_suffix = ''
        stcs_epochs_roi_fname = op.join(stcs_epochs_path, f'stcs_epochs_{label_name}'+orientation_fname_suffix+resample_fname_suffix)

        if op.exists(stcs_epochs_roi_fname):
            print(f'{subject} stcs {label_name} already exists. Skipping.')
        else:
            print(f'{subject} stcs {label_name} not found. Computing.')
            os.makedirs(stcs_epochs_path, exist_ok=True)
        
            log_fname = op.join(config.logs_dir, f'{subject}_logfile.csv')
            epochs_path = op.join(preprocessed_data_path, subject, 'epoch')
            epochs_fname = op.join(epochs_path, f'{subject}_epo.fif')
            inv_fname = op.join(epochs_path, f'{subject}_{ch_type}_inv.fif')
            epochs_log_fname = op.join(stcs_epochs_path, 'epochs_matched_logfile.csv')

            epochs = mne.read_epochs(epochs_fname, preload=True, verbose=False) # read in epochs
            epochs.crop(tmin=-0.2, tmax=1.4)
            if resample:
                epochs.resample(100)
            print('Number of epochs: ', len(epochs.events))
            trial_info = pd.read_csv(log_fname) # get subject logfile

            # get epochs drop log as a mask, then apply it to trial info as epochs.metadata
            if not op.exists(epochs_log_fname):
                epochs_drop_mask = [not bool(epoch) for epoch in epochs.drop_log]
                assert(epochs_drop_mask.count(True) == len(epochs.events)) # sanity
                assert(len(trial_info[epochs_drop_mask]) == len(epochs.events))
                print('Saving epochs-matched logfile to disk.')
                trial_info[epochs_drop_mask].to_csv(epochs_log_fname)

            # read in inverse operator
            inv = mne.minimum_norm.read_inverse_operator(inv_fname, verbose=False)
            snr = 2.0 # SNR assumption for evoked; for epoch use 2
            lambda2 = 1.0 / snr ** 2
            method = 'MNE'

            print(f'Making stcs ({label_name}).')
            stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2, label=label, method=method, verbose=False)
            np.save(file=stcs_epochs_roi_fname, arr=stcs)
            
    print(f'{subject} done.')
