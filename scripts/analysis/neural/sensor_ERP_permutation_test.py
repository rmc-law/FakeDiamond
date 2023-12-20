#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:17:58 2023

@author: rl05

Analysis - sensor space, N400, cluster-based permutation test
"""

import os
import os.path as op
import sys
import pickle

import numpy as np
# import pandas as pd

from mne import read_epochs, read_source_estimate
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.viz import plot_compare_evokeds

from eelbrain import Dataset, load, Factor, plot, testnd

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')
results_dir = op.join(config.project_repo, 'results')

preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

subjects_to_ignore = [
    '17',
    '24',
    '25',
    '26'
]

subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

# ch_type = input('MEEG or MEG: ')
# experiment = input('compose or specificity: ')
# parc = 'fake_diamond'



# X = # 

# t_obs, clusters, cluster_pv, _ = spatio_temporal_cluster_1samp_test(X, threshold=None, n_permutations=1024, tail=0, stat_fun=None, adjacency=None, n_jobs=None, seed=None, max_step=1, spatial_exclude=None, step_down_p=0, t_power=1, out_type='indices', check_disjoint=False, buffer_size=1000, verbose=None)[source]

# =============================================================================
# Prepare eelbrain dataset
# =============================================================================

# # get condition names
# epoch_path = op.join(preprocessed_data_path, 'sub-01', 'epoch')
# epoch_fname = op.join(epoch_path, 'sub-01_epo.fif')
# epochs = read_epochs(epoch_fname, preload=False, verbose=False)
# conditions = list(epochs.event_id.keys())
# del epochs

# read in epochs 
epochs_data = []
subject_list = []
condition_list = []
concreteness_list = []
denotation_list = []

denotation = ['baseline','subsective','privative']
concreteness = ['concrete','abstract']

for subject in subjects:
    subject = f'sub-{subject}'
    print(f'Reading in epochs from {subject}.')
    epochs_path = op.join(data_dir, 'preprocessed', subject, 'epoch', f'{subject}_epo.fif')
    epochs = read_epochs(epochs_path, verbose=False).crop(None, 0.6)
    epochs.equalize_event_counts()
    for c in concreteness:
        for d in denotation:
            e = epochs[f'{c}/{d}'].copy()
            e = load.mne.epochs_ndvar(e,data='eeg')
            epochs_data.append(e)
            subject_list.append(subject)
            concreteness_list.append(c)
            denotation_list.append(d)
            condition_list.append(f'{c}-{d}')

ds = Dataset()

ds['epochs'] = epochs_data 
ds['subject'] = Factor(subject_list,random=True)

ds['condition'] = Factor(condition_list)
ds['condition'].sort_cells(['concrete-baseline',
                            'concrete-subsective',
                            'concrete-privative',
                            'abstract-baseline',
                            'abstract-subsective',
                            'abstract-privative'         
                            ])
ds['concreteness'] = Factor(concreteness_list)
ds['concreteness'].sort_cells(['concrete','abstract'])
ds['denotation'] = Factor(denotation_list)
ds['denotation'].sort_cells(['baseline','subsective','privative'])


# =============================================================================
# MNE implementation
# =============================================================================

cluster_stats = dict.fromkeys(['grad', 'mag', 'eeg'])

# initialise cluster stats objects
for key in cluster_stats.keys():
    cluster_stats[key] = dict.fromkeys(['concreteness', 'denotation'])
    

for ch_type in ['grad', 'mag', 'eeg']:
    evokeds = dict.fromkeys(['concrete','abstract',
                             'baseline','subsective','privative'])

    for key in evokeds.keys():
        evokeds[key] = list()

    for condition in evokeds.keys():
        for subject in subjects:
            subject = f'sub-{subject}'
            print(f'Reading in epochs from {subject}.')
            epochs_path = op.join(data_dir, 'preprocessed', subject, 'epoch', f'{subject}_epo.fif')
            evoked = mne.read_evokeds(path.join(data_path, f"{sbj_id}_{condition}_evokeds_-ave.fif"))[0]
            evoked.resample(250)
            if (sbj_id==12) & (ch_type=='eeg'):
                pass
            else:                
                evokeds[condition].append(evoked.get_data(picks=ch_type))