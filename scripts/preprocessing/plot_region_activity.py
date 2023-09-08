#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:01:44 2023

@author: rl05
"""


import os
import os.path as op

from mne import (read_epochs, read_source_spaces, read_labels_from_annot,
                 read_source_estimate, read_label)
# from mne.viz import Brain
import matplotlib.pyplot as plt
import numpy as np

import config 
# from helper import calculate_avg_sem

subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')
figs_dir = op.join(config.project_repo, 'figures')

preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

fsaverage_src_fname = op.join(subjects_dir, 'fsaverage_src', 'fsaverage_src.fif')
src_fsaverage = read_source_spaces(fsaverage_src_fname)

subjects = config.subject_ids
subjects_to_ignore = ['05', # bem problem
                      '07', # no mri
                      '11' # no mri
                      ]
subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

epoch_path = op.join(preprocessed_data_path, 'sub-01', 'epoch')
epoch_fname = op.join(epoch_path, 'sub-01_epo.fif')
epochs = read_epochs(epoch_fname, preload=False)
conditions = list(epochs.event_id.keys())
del epochs

