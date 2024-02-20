#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:16:38 2024

@author: rl05
"""


import os
import os.path as op
import sys
import pickle

import numpy as np
# import pandas as pd

from mne import read_source_spaces, read_source_estimate#, spatial_src_adjacency
# from mne.stats import (
#     f_oneway,
#     f_mway_rm,
#     f_threshold_mway_rm,
#     spatio_temporal_cluster_test,
#     summarize_clusters_stc,
# )
from eelbrain import Dataset, load, Factor, plot, testnd

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')
results_dir = op.join(config.project_repo, 'results')

preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

analysis = 'roi'
output = op.join(results_dir, 'neural', analysis)

subjects_to_ignore = [
                      ]

subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

ch_type = input('MEEG or MEG: ')
experiment = input('analysis? (compose, concreteness, denotation, specificity): ')
parc = 'fake_diamond'

if experiment == 'compose' or experiment == 'concreteness' or experiment == 'denotation':
    conditions = ['concrete-subsective','concrete-privative','concrete-baseline',
                  'abstract-subsective','abstract-privative','abstract-baseline']
elif experiment == 'specificity':
    conditions = ['low','mid','high']


src_fname = op.join(data_dir, 'mri', 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src = read_source_spaces(src_fname) 
fsave_vertices = [src[0]["vertno"], src[1]["vertno"]]


# create eelbrain dataset

stcs = []
subject_list = []
condition_list = []

for subject in subjects:
    subject = f'sub-{subject}'
    print(f'Reading in source estimates from {subject}.')
    stc_path = op.join(data_dir, 'stcs', subject)
    for condition in conditions:
        condition = condition.replace('/','-')
        stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}-lh.stc'),
                                    subject='fsaverage')
        stcs.append(stc)
        subject_list.append(subject)
        condition_list.append(condition)

ds = Dataset()

if experiment == 'compose' or experiment == 'concreteness' or experiment == 'denotation':
    condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    denotation = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
elif experiment == 'specificity':
    specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]

ds['stcs'] = load.fiff.stc_ndvar(stcs, 
                                  subject='fsaverage_src', 
                                  src='oct-6', 
                                  parc=parc) 
ds['subject'] = Factor(subject_list,random=True)

if experiment == 'compose' or experiment == 'concreteness' or experiment == 'denotation':    
    ds['condition'] = Factor(condition)
    ds['condition'].sort_cells(['concrete-baseline',
                                'concrete-subsective',
                                'concrete-privative',
                                'abstract-baseline',
                                'abstract-subsective',
                                'abstract-privative'         
                                ])
    ds['concreteness'] = Factor(concreteness)
    ds['concreteness'].sort_cells(['concrete','abstract'])
    ds['denotation'] = Factor(denotation)
    ds['denotation'].sort_cells(['baseline','subsective','privative'])
elif experiment == 'specificity':
    ds['specificity'] = Factor(specificity)
    ds['specificity'].sort_cells(['low','mid','high'])
stc_reset = ds['stcs']


'''.............................. run ROI test ..............................'''

rois = ['anteriortemporal-lh', 'posteriortemporal-lh',
        'inferiorfrontal-lh', 'temporoparietal-lh']

for roi in rois:
    ds['stcs'] = stc_reset
    stc_region = stc_reset.sub(source=roi) # subset language network region data
    ds['stcs'] = stc_region # assign this back to the ds

    # perform temporal permutation test in a particular region
    if experiment == 'compose':
        x = 'concreteness*denotation*subject'
    elif experiment == 'concreteness':
        x = 'concreteness*subject'
    elif experiment == 'denotation':
        x = 'denotation*subject'
    elif experiment == 'specificity':
        x = 'specificity*subject'
    res = testnd.ANOVA(y=ds['stcs'].mean('source'), 
                        x=x, 
                        data=ds, 
                        samples=5000, 
                        # pmin=0.05,
                        tstart=0.6,
                        tstop=1.4,
                        tfce=True,
                        match='subject')

    pickle.dump(res, open(op.join(results_dir, f'{roi}.pickle'), 'wb'))

    f = open(op.join(output, f'{experiment}_{ch_type}_{roi}_results_table.txt'), 'w')
    f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
    f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s, mintime=??\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
    f.write(str(res.clusters))
    f.close()

    pmin = 0.05
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_nb = i+1
            cluster = sign_clusters[i]['cluster']
            tstart = sign_clusters[i]['tstart']
            tstop = sign_clusters[i]['tstop']
            effect = sign_clusters[i]['effect']

            effect = effect.replace(' x ', '%') # Changing format of string for interaction effects.

            print('Plotting time series for %s' %roi)
            timecourse = stc_region.mean('source')
            activation = plot.UTSStat(timecourse, 
                                      effect, 
                                      ds=ds, 
                                      error='sem', 
                                      match='subject', 
                                      legend='lower left', 
                                      xlabel='Time (ms)', 
                                      ylabel='Activation (MNE)', 
                                      xlim=(0.6,1.4), 
                                      title=f'Cluster {i+1}: Effect of {effect} at {roi}')
            activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
            # activation.add_vline(0, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(3, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(3.6, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(4.2, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation._axes[0].set_xticks([0,tstart,tstop])

            activation.save(op.join(output, f'fig_{experiment}_{ch_type}_{roi}_cluster{i+1}_timecourse_{effect}_({tstart}-{tstop})_{roi}.png'))
            activation.close()

            ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
            bar = plot.Barplot(ds['average_source_activation'], effect, ds=ds, title=f'Cluster {i+1}, {tstart}-{tstop}, {roi}', match='subject', ylabel='Average source activation (dSPM)')
            bar.save(op.join(output, f'fig_{experiment}_{roi}_cluster{i+1}_bar_{effect}_({tstart}-{tstop}).png'))
            bar.close()
