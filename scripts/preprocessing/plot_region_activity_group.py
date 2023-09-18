#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:01:44 2023

@author: rl05
"""

import os
import os.path as op

import numpy as np
import pandas as pd
import pickle 

import matplotlib.pyplot as plt

from mne import read_source_spaces, read_epochs, read_labels_from_annot, read_source_estimate
from eelbrain import Dataset, load, Factor, plot
# from surfer import Brain

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
                      '17' # not yet processed
                      ]
subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

epoch_path = op.join(preprocessed_data_path, 'sub-01', 'epoch')
epoch_fname = op.join(epoch_path, 'sub-01_epo.fif')
epochs = read_epochs(epoch_fname, preload=False)
conditions_all = list(epochs.event_id.keys())
del epochs

ch_type = 'MEEG'

experiments = ['compose', 'specificity']
parcs = ['fake_diamond', 'ventral_ATL']

for experiment in experiments: 
        
    for parc in parcs:
        if parc == 'fake_diamond':
            annot = read_labels_from_annot('fsaverage_src', parc=parc, hemi='lh')[:-1]
        elif parc == 'ventral_ATL':
            annot = read_labels_from_annot('fsaverage_src', parc=parc, hemi='both')[:-2]
        
        if experiment == 'compose':
            conditions_subset = conditions_all[:6]
        if experiment == 'specificity':
            conditions_subset = conditions_all[6:]
        
        stcs, subject_list, condition_list = [], [], []
        print('Read in stcs from subjects.')
        for subject in subjects:
            subject = f'sub-{subject}'
            for condition in conditions_subset:
                condition = condition.replace('/','-')
                stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
                stc = read_source_estimate(stc_fname)
                stcs.append(stc)
                subject_list.append(subject)
                condition_list.append(condition)
                del stc
        
        
        print ('Create eelbrain dataset.')
        ds = Dataset()
        
        if experiment == 'compose':
            concreteness = [condition.split('-')[0] for condition in condition_list]
            denotation = [condition.split('-')[1] for condition in condition_list]
        elif experiment == 'specificity':
            specificity = [condition.split('-')[0] for condition in condition_list]
        
        ds['stcs'] = load.fiff.stc_ndvar(stcs, 
                                         subject='fsaverage_src', 
                                         src='oct-6', 
                                         parc=parc) 
        ds['subject'] = Factor(subject_list,random=True)
        ds['condition'] = Factor(condition_list)
        
        if experiment == 'compose':    
            ds['concreteness'] = Factor(concreteness)
            ds['denotation'] = Factor(denotation)
        elif experiment == 'specificity':
            ds['specificity'] = Factor(specificity)
            ds['specificity'].sort_cells(['low','mid','high'])
        src_reset = ds['stcs']
        
        
        
        
        '''...................... plot time courses by region .......................'''
        
        regions = [label.name for label in annot]
        
        if experiment == 'compose':
            groupings = ['concreteness', 'denotation', 'condition']
            
            color_maps = [# orange and purple for concrete vs. abstract
                          np.vstack([plt.cm.Purples(0.7),plt.cm.Oranges(0.7)]).tolist(), 
                          
                          # grey for denotation
                          plt.cm.Greys([0.3, 0.6, 0.9]).tolist(), 
                          
                          # oranges and purples modulated by brightness for full conditions
                          np.vstack([plt.cm.Purples([0.3, 0.6, 0.9]),
                                     plt.cm.Oranges([0.3, 0.6, 0.9])]).tolist()
                          
                          ]
        
        elif experiment == 'specificity':
            groupings = ['specificity']
            color_maps = [plt.cm.Greens([0.3, 0.6, 0.9]).tolist()]
        
        # group together conditions and colour maps
        for grouping, colors in zip(groupings, color_maps):
            
            for region in regions:
            
                ds['srcm'] = src_reset
                src_region = src_reset.sub(source=region) # subset language network region data
                ds['srcm'] = src_region # assign this back to the ds
                timecourse = src_region.mean('source')
            
                activation = plot.UTSStat(timecourse, grouping, ds=ds, 
                                          # error='sem', 
                                          error=None,
                                          match='subject', 
                                          legend=None, 
                                          xlabel='Time (ms)', 
                                          ylabel='Activation (MNE)', 
                                          tight=True, 
                                          xlim=(-0.5, 1.4), 
                                          colors=colors,
                                          frame=None, 
                                          title=f'Time series at {region}')
                activation.axes[0].set_xticks([-0.3, 0, 0.6, 1.4])
                activation.axes[0].axvline(-0.3, lw=1, color='black')
                activation.axes[0].axvline(0, lw=1, color='black')
                activation.axes[0].axvline(0.6, lw=1, color='black')
                # activation.axes[0].lines[0].set_lw(w=2)
                # activation.axes[0].lines[1].set_ls(ls='--')
                # activation.axes[0].lines[1].set_lw(w=2)
                # activation._axes[0].legend(labels=['list-inside-sentence','list-inside-list'], loc='upper right')
                # activation.set_ylim(1.15,3)
                # activation._axes[0].set_xticks([0,0.6])
                activation.figure.set_size_inches(w=8, h=4)
                activation.save(op.join(figs_dir, 'group', f'{grouping}_{region}_MEEG_group(n={len(subjects)}).png'), 
                                dpi=100)
                activation.close()
        
        # @ colour scheme
        # position  (5,6,7): '#DDAA33','#AA3377','#004488'
        # association (high, low): '#CCBB44','#AA4499'
        # composition (sent, list): '#332288','#117733'
        # composition*position: '#EE7733','#CC3311','#EE3377','#0077BB', '#33BBEE','#009988'
        # condition (high_sent, high_list, low_sent, low_list): 'royalblue', 'orange', 'lightseagreen', 'tomato'