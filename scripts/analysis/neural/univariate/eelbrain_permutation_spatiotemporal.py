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
from eelbrain import Dataset, load, Factor, plot, testnd, set_parc

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = config.subject_ids
print(f'subjects (n={len(subjects)}): ', subjects)

decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
data_dir = op.join(config.project_repo, 'data')
results_dir = op.join(config.project_repo, 'results')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

analysis_type = input('anova or lm: ')
ch_type = 'MEEG'
analysis = input('analysis? (composition, concreteness, denotation, specificity): ')
parc = 'fake_diamond'
data_type = input('mne or decod: ')


if analysis == 'composition' or analysis == 'concreteness' or analysis == 'denotation':
    conditions = ['concrete-subsective','concrete-privative','concrete-baseline',
                  'abstract-subsective','abstract-privative','abstract-baseline']
elif analysis == 'specificity':
    conditions = ['low','mid','high']


src_fname = op.join(data_dir, 'mri', 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src = read_source_spaces(src_fname) 
fsave_vertices = [src[0]["vertno"], src[1]["vertno"]]


#%% make eelbrain dataset

stcs = []
subject_list = []
condition_list = []

for subject in subjects:
    subject = f'sub-{subject}'
    print(f'Reading in source estimates from {subject}.')
    stc_path = op.join(data_dir, 'stcs', subject)
    for condition in conditions:
        condition = condition.replace('/','-')
        if data_type == 'mne':
            stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}-lh.stc'), subject='fsaverage')
        elif data_type == 'decod':
            if (subject == 'sub-12') or (subject == 'sub-13'):
                continue
            coef_dir = op.join(decoding_dir, f'output/{analysis}/diagonal/logistic/MEEG/{subject}')            
            coef_projection_fname = op.join(coef_dir, f'source_projection_coef_{analysis}-lh.stc')
            stc = read_source_estimate(coef_projection_fname, subject='fsaverage')
        stcs.append(stc)
        subject_list.append(subject)
        condition_list.append(condition)


ds = Dataset()

if analysis == 'composition' or analysis == 'concreteness' or analysis == 'denotation':
    condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    denotation = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
elif analysis == 'specificity':
    specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]

ds['stcs'] = load.fiff.stc_ndvar(stcs, 
                                  subject='fsaverage_src', 
                                  src='oct-6', 
                                  parc=parc) 
ds['subject'] = Factor(subject_list,random=True)

if analysis == 'composition' or analysis == 'concreteness' or analysis == 'denotation':    
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
elif analysis == 'specificity':
    ds['specificity'] = Factor(specificity)
    ds['specificity'].sort_cells(['low','mid','high'])
stc_reset = ds['stcs']


stc_reset = set_parc(stc_reset, 'left_hemi')
stc_region = stc_reset.sub(source='left_hemi-lh')
ds['stcs'] = stc_region
region = ['left_hemi-lh']


#%% run roi test


# perform temporal permutation test in a particular region        
if analysis_type == 'anova':
    output_dir = op.join(results_dir, 'neural/left_hemi/anova', analysis)
    if not op.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if analysis == 'composition':
        x = 'concreteness*denotation*subject'
    elif analysis == 'concreteness':
        x = 'concreteness*subject'
    elif analysis == 'denotation':
        x = 'denotation*subject'
    elif analysis == 'specificity':
        x = 'specificity*subject'
            
    res = testnd.ANOVA(y=ds['stcs'].mean('source'), 
                        x=x, 
                        data=ds, 
                        samples=1000, 
                        pmin=0.05,
                        # tstart=0.8,
                        # tstop=1.2,
                        tstart=0.2,
                        tstop=0.6,
                        # tfce=True,
                        match='subject')
    pickle.dump(res, open(op.join(output_dir, 'spatiotemporal_left_hemi.pickle'), 'wb'))

    f = open(op.join(output_dir, f'{analysis}_{ch_type}_spatiotemporal_results_table.txt'), 'w')
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
    
            # save significant cluster as a label for plotting.
            # print('Saving clusters as labels...')
            # label = labels_from_clusters(cluster)
            # label[0].name = 'label-lh'
            # mne.write_labels_to_annot(label, subject='fsaverage', parc='cluster%s_FullAnalysis'%i, subjects_dir=subjects_dir,  overwrite=True)
            # stc = set_parc(stc, 'cluster%s_FullAnalysis' %i)
            # stc_region = stc.sub(source='label-lh')
            # ds['stcs'] = stc_region
    
            print('Plotting time series for %s' %region)
            timecourse = stc_region.mean('source')
            activation = plot.UTSStat(timecourse, effect, ds=ds, error='sem', match='Subject', legend='lower left', xlabel='Time (ms)', ylabel='Activation (dSPM)', xlim=(0,0.6), title='Cluster %s: Effect of %s at %s' %(i+1, effect, region))
            activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
            activation._axes[0].set_xticks([0,tstart,tstop])
            activation.save(op.join(output_dir, 'cluster%s_(%s-%s)_%s_%s.png' %(i+1,tstart, tstop, effect, region)), dpi=300)
            activation.close()
    
            ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop))
            bar = plot.Barplot(ds['average_source_activation'], effect, ds=ds, title='Average activation at %s' %region, match='Subject', ylabel='Average source activation (dSPM)')
            bar.save(op.join(output_dir, 'cluster%s_Bar_%s_(%s-%s).png'%(i+1, effect, tstart, tstop)))
            bar.close()
            print('Done plotting for %s. \n' %region)
    
            brain = plot.brain.cluster(cluster.mean('time'), surf='inflated', hemi='lh', colorbar=True, time_label='ms', w=600, h=400, foreground='white', background='black', subjects_dir=subjects_dir)
            brain.save_image(op.join(output_dir, 'cluster%s_brain_(%s-%s)_%s.png' %(i+1, tstart, tstop, effect)))
            # brain.close()
            
elif analysis_type == 'lm':
    output_dir = op.join(results_dir, 'neural/roi/lm', roi, analysis)
    if not op.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if analysis == 'composition':
        x = 'concreteness*denotation'
    elif analysis == 'specificity':
        x = 'specificity'
        
    lms = []
    for subject in subjects:
        
        stcs = []
        subject_list = []
        condition_list = []

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
        
        if analysis == 'composition' or analysis == 'concreteness' or analysis == 'denotation':
            condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
            concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
            denotation = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
        elif analysis == 'specificity':
            specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]
        
        ds['stcs'] = load.fiff.stc_ndvar(stcs, 
                                         subject='fsaverage_src', 
                                         src='oct-6', 
                                         parc=parc) 
        ds['subject'] = Factor(subject_list)
        
        if analysis == 'composition' or analysis == 'concreteness' or analysis == 'denotation':    
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
        elif analysis == 'specificity':
            ds['specificity'] = Factor(specificity)
            ds['specificity'].sort_cells(['low','mid','high'])
        # stc_reset = ds['stcs']

        print(f'fitting lm to {subject} data.')
        lm = testnd.LM(y=ds['stcs'].mean('source'), 
                       x='concreteness', 
                       data=ds, 
                       samples=0, 
                       # pmin=0.05,
                       tstart=0.6,
                       tstop=1.4,
                       # tfce=True,
                       subject=subject
                       )
        lms.append(lm)
        
    rows = []
    for lm in lms:
        rows.append([lm.subject, lm.t('intercept'), lm.t('concreteness')])#, lm.t('denotation')])#, lm.t('concreteness x denotation')])
    # When creating the dataset for stage 2 analysis, declare subject as random factor;
    # this is only relevant if performing ANOVA as stage 2 test.
    data = Dataset.from_caselist(['subject', 'intercept', 'concreteness'], rows, random='subject')
    data
    # pickle.dump(res, open(op.join(output_dir, f'{roi}.pickle'), 'wb'))

        
    result = testnd.TTestOneSample('concreteness', data=data, pmin=0.05, tstart=0.6, tstop=1.4)
    p = plot.UTS(result)#, t=[0.120, 0.155, None], title=result, head_radius=0.35)
    
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
            
    p_cb = p.plot_colorbar(right_of=p.axes[0], label='t')