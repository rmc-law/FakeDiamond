#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:34:39 2023

@author: rl05
"""

import os
import os.path as op

project = 'fake_diamond'

cbu_repo_meg = op.join('/megdata/cbu', project)
project_repo = op.join('/imaging/hauk/rl05/', project)
bids_root = op.join(project_repo, 'data')
# data_dir = op.join(project_repo, 'data', 'meg')
# subjects_dir = op.join(project_repo, 'data', 'mri')
logs_dir = op.join(project_repo, 'logs')

if not op.isdir(project_repo):  
    os.mkdir(project_repo)

# maxwell filtering files
calibration_fname = '/neuro_triux/databases/sss/sss_cal.dat'
crosstalk_fname = '/neuro_triux/databases/ctc/ct_sparse.fif'

# =============================================================================
# Mapping between subject and filename
# =============================================================================

subject_ids = ['01','02']

map_subjects_meg = {
        '01': ('meg23_285', '230807'),
        '02': ('meg23_293', '230810')
    }

# subject names of MRI data
map_subjects_mri = {
        '01': ('CBU230541'),
        '02': ('CBU210688')
    }

runs = 5


# =============================================================================
# Bad and flat channels
# =============================================================================

bad_chs = {
        '01': {'eeg': ['EEG001','EEG002','EEG005'],
               'meg': []},
        '02': {'eeg': [],
               'meg': []}
    }


# head origins

head_origins = {
        '01': [(9.4, 8.9, -44.9),(7.7, 10.4, -47.9),(8.2, 10.6, -49.8),(9.8, 10.8, -50), (9.3, 9.8, -52.1)],
        '02': [(1.7, 0.8, -56.4),(3.6, 15.4, -56.7),(1.1, 15.9, -56.0),(2.4, 13.0, -57.2),(3.7, 13.2, -57.6)]
    }


# =============================================================================
# Digital filtering parameters
# =============================================================================

l_freq = 0.1
h_freq = 40

event_id = {
    'concrete/subsective/word1':111, 
    'concrete/subsective/word2':112,
    'concrete/privative/word1':121, 
    'concrete/privative/word2':122,
    'concrete/baseline/word1':131, 
    'concrete/baseline/word2':132,
    'abstract/subsective/word1':211, 
    'abstract/subsective/word2':212,
    'abstract/privative/word1':221, 
    'abstract/privative/word2':222,
    'abstract/baseline/word1':231, 
    'abstract/baseline/word2':232,
    'low/word1':71, 
    'low/word2':72,
    'mid/word1':81, 
    'mid/word2':82,
    'high/word1':91, 
    'high/word2':92,
    'button/left':4096,
    'button/right':8192,
    'photodiode':512
}
