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

stc_path = op.join(data_dir, 'stcs')

subjects = config.subject_ids
subjects_to_ignore = [
                      '07', # no mri
                      '13', # not yet processed
                      '14', # not yet processed
                      '17' # not yet processed
                      ]
subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

epoch_path = op.join(preprocessed_data_path, 'sub-01', 'epoch')
epoch_fname = op.join(epoch_path, 'sub-01_epo.fif')
epochs = read_epochs(epoch_fname, preload=False)
conditions = list(epochs.event_id.keys())
del epochs

ch_type = 'MEEG'

# read in labels
annot = read_labels_from_annot('fsaverage_src', parc='fake_diamond', hemi='lh')[:-1]


# =============================================================================
# visualise individual subjects
# =============================================================================

colors = plt.cm.Purples(np.linspace(0.2, 1, len(conditions)))
        
for label in annot:
    
    label.subject = 'fsaverage'
    
    number_of_subplots = len(subjects)
    scaling_factor = 1.5
    fig_height = scaling_factor * number_of_subplots
    fig, axes = plt.subplots(number_of_subplots, sharex=True)
    fig.set_size_inches(16, fig_height)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(f'Activity in {label.name} in individual subjects')
    
    for subject, axis in zip(subjects, axes): 
    
       subject = f'sub-{subject}'
    
       for condition, color in zip(conditions, colors):
        
            condition = condition.replace('/','-')
            
            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
            stc = read_source_estimate(stc_fname)
            time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
            
            axis.plot(stc.times, time_course, color=color, alpha=0.9)
            axis.set_title(subject)
            
    fig_dir = op.join(config.project_repo, 'figures', 'individual_subjects')
    fig.savefig(op.join(fig_dir, f'{label.name}_MEEG_ind_subjs.png'))
    plt.close()
    
    
# =============================================================================
# analysis composition
# =============================================================================

times = np.linspace(-0.5, 1.4, 475)
colors = plt.cm.Purples(np.linspace(0.2, 1, len(conditions)))
        
number_of_subplots = len(subjects)
scaling_factor = 1.5
fig_height = scaling_factor * number_of_subplots
fig, axes = plt.subplots(number_of_subplots, sharex=True)
fig.set_size_inches(16, fig_height)
fig.subplots_adjust(hspace=0.4)
fig.suptitle(f'Activity in {label.name} in individual subjects')
    
for label, axis in (annot, axes):
    
    label.subject = 'fsaverage'

    for i_subject, subject in enumerate(subjects): 
    
       subject = f'sub-{subject}'
    
       group_average = np.zeros(len(subjects), len(conditions), times)
       
       for j_condition, (condition, color) in enumerate(zip(conditions, colors)):
        
            condition = condition.replace('/','-')
            
            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
            stc = read_source_estimate(stc_fname)
            time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
            
            group_average[i_subject, j_condition, :] = time_course
            
            
    axis.plot(stc.times, time_course, color=color, alpha=0.9)
    axis.set_title(subject)
            
    fig_dir = op.join(config.project_repo, 'figures', 'individual_subjects')
    fig.savefig(op.join(fig_dir, f'{label.name}_MEEG_ind_subjs.png'))
    plt.close()