#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:13:01 2023

@author: rl05
"""

import os
import os.path as op
import sys
import pickle

import numpy as np
# import pandas as pd

from mne import read_epochs, read_source_estimate
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
    '24'
]

subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

ch_type = input('MEEG or MEG: ')
experiment = input('compose or specificity: ')
parc = 'fake_diamond'

# get condition names
epoch_path = op.join(preprocessed_data_path, 'sub-01', 'epoch')
epoch_fname = op.join(epoch_path, 'sub-01_epo.fif')
epochs = read_epochs(epoch_fname, preload=False, verbose=False)
conditions = list(epochs.event_id.keys())
del epochs

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

if experiment == 'compose':
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

if experiment == 'compose':    
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
    res = testnd.anova(y=ds['stcs'].mean('source'), 
                       x='concreteness*denotation*Subject', 
                       data=ds, 
                       samples=5000, 
                       pmin=0.05,
                       tfce=True,
                       match='Subject')

    pickle.dump(res, open(op.join(results_dir, f'{roi}.pickle'), 'wb'))

    f = open(op.join(results_dir, f'{roi}_results_table.txt'), 'w')
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

            print('Plotting time series for %s' %region)
            timecourse = src_region.mean('source')
            activation = eelbrain.plot.UTSStat(timecourse, effect, ds=ds, error='sem', match='Subject', legend='lower left', xlabel='Time (ms)', ylabel='Activation (dSPM)', xlim=(0,0.6), title='Cluster %s: Effect of %s at %s' %(i+1, effect, region))
            activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
            # activation.add_vline(0, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(3, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(3.6, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            # activation.add_vline(4.2, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
            activation._axes[0].set_xticks([0,tstart,tstop])

            activation.save(os.path.join(output, 'clus%s_%s_%s_(%s-%s).png' %(i+1, tstart, tstop, effect, region)), dpi=250)
            activation.close()

            ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
            bar = plot.Barplot(ds['average_source_activation'], effect, ds=ds, title='Average activation at %s' %region, match='Subject', ylabel='Average source activation (dSPM)')
            bar.save(os.path.join(output, 'cluster%s_BarGraph_(%s-%s)_effect=%s.png'%(i+1, tstart, tstop, effect)))
            bar.close()
