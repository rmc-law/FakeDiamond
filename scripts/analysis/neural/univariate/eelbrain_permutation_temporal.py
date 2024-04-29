#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:16:38 2024

@author: rl05

Perform cluster-based permutation tests on ROI data
"""


import os
import os.path as op
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from mne import read_source_estimate
from eelbrain import Dataset, load, Factor, plot, testnd, test
from eelbrain._stats.stats import variability

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = config.subject_ids
print(f'subjects (n={len(subjects)}): ', subjects)

data_dir = op.join(config.project_repo, 'data')
results_dir = op.join(config.project_repo, 'results')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

analysis_type = 'anova'#input('anova or lm: ')
ch_type = 'MEEG'
analysis = input('analysis? (replicate, composition, denotation, specificity): ')
parc = 'fake_diamond'

#%% figure style -- not mandatory
poster_style = {
    'font.size': 20,       # Increase font size for titles and labels
    'axes.titlesize': 20,  # Increase title font size
    'axes.labelsize': 18,  # Increase label font size
    'figure.figsize': (10, 8),  # Set figure size (width, height)
    'axes.edgecolor': 'grey',  # Set axis color
    'axes.linewidth': 3,      # Increase axis linewidth
    'axes.grid': False,        # Show grid lines
    'grid.color': 'lightgray', # Set grid color
    'lines.linewidth': 3,     # Increase line thickness
    'xtick.labelsize': 18,    # Increase x-axis tick label size
    'ytick.labelsize': 18,    # Increase y-axis tick label size
    'xtick.color': 'grey',
    'ytick.color': 'grey',
    'xtick.major.size': 9,
    'ytick.major.size': 9,
    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'legend.fontsize': 14,    # Increase legend font size
    'legend.frameon': True,   # Display legend frame
    'legend.edgecolor': 'black'  # Legend frame color
}

#%% make eelbrain dataset

stcs = []
subject_list = []
condition_list = []
ds = Dataset()

if analysis == 'composition' or analysis == 'replicate':
    conditions = ['concrete-baseline','concrete-subsective','abstract-baseline','abstract-subsective']
    # concreteness = ['concrete','abstract']
    # composition = ['word','phrase']
    # for c in concreteness:
    #     for d in composition:
    #         condition = f'{c}-{d}'
    #         for subject in subjects:
    #             subject = f'sub-{subject}'
    #             stc_path = op.join(data_dir, 'stcs', subject)
    #             if composition == 'word':
    #                 stc = read_source_estimate(op.join(stc_path, f'{subject}_{c}-baseline_{ch_type}-lh.stc'), subject='fsaverage_src')
    #             else:
    #                 stcs_tmp = []
    #                 for d in ['subsective','privative']:
    #                     stc = read_source_estimate(op.join(stc_path, f'{subject}_{c}-{d}_{ch_type}-lh.stc'), subject='fsaverage_src')
    #                     stcs_tmp.append(stc)
    #                 stcs_tmp = [stc._data for stc in stcs_tmp]
    #                 stcs_tmp = np.array([np.mean(x) for x in zip(*stcs_tmp)])
    #                 stc._data = stcs_tmp
    #             stcs.append(stc)
    #             subject_list.append(subject)
    #             condition_list.append(condition)
if analysis == 'denotation':
    conditions = ['concrete-subsective','concrete-privative','abstract-subsective','abstract-privative']
elif analysis == 'specificity':
    conditions = ['low','mid','high']
for subject in subjects:
    subject = f'sub-{subject}'
    print(f'Reading in source estimates from {subject}.')
    stc_path = op.join(data_dir, 'stcs', subject)
    for condition in conditions:
        condition = condition.replace('/','-')
        stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}-lh.stc'), subject='fsaverage_src')
        stcs.append(stc)
        subject_list.append(subject)
        condition_list.append(condition)

if analysis == 'composition' or analysis == 'replicate':
    condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    composition = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
elif analysis == 'denotation':
    condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    denotation = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
elif analysis == 'specificity':
    specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]

ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc=parc) 
ds['subject'] = Factor(subject_list,random=True)

if analysis == 'composition' or analysis == 'replicate':
    ds['condition'] = Factor(condition)
    ds['condition'].sort_cells(['concrete-baseline',
                                'concrete-subsective',
                                'abstract-baseline',
                                'abstract-subsective'])
    ds['concreteness'] = Factor(concreteness)
    ds['concreteness'].sort_cells(['concrete','abstract'])
    ds['composition'] = Factor(composition)
    ds['composition'].sort_cells(['baseline','subsective'])
elif analysis == 'denotation':
    ds['condition'] = Factor(condition)
    ds['condition'].sort_cells(['concrete-subsective',
                                'concrete-privative',
                                'abstract-subsective',
                                'abstract-privative'         
                                ])
    ds['concreteness'] = Factor(concreteness)
    ds['concreteness'].sort_cells(['concrete','abstract'])
    ds['denotation'] = Factor(denotation)
    ds['denotation'].sort_cells(['subsective','privative'])
elif analysis == 'specificity':
    ds['specificity'] = Factor(specificity)
    ds['specificity'].sort_cells(['low','mid','high'])
stc_reset = ds['stcs']


#%% run roi test

rois = ['anteriortemporal-lh', 'posteriortemporal-lh','inferiorfrontal-lh', 'temporoparietal-lh']#, 'lateraloccipital-lh']

# this is to replicate Bemis & Pylkkanen 2011
plt.rcParams.update(poster_style)

if analysis == 'replicate':
    roi = 'anteriortemporal-lh'
    ds['stcs'] = stc_reset
    stc_region = stc_reset.sub(source=roi) # subset language network region data
    ds['stcs'] = stc_region # assign this back to the ds
    output_dir = op.join(results_dir, 'neural/roi/anova', roi, 'replicateB&P')
    if not op.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    x = 'concreteness*composition*subject'
    result = test.ANOVA(y=ds['stcs'].mean('source').mean(time=(0.8,0.85)), x=x, data=ds)
    print(result, file=open(op.join(output_dir, 'anova.txt'), 'w'))
    
    roi_activity = stc_region.mean('source').mean(time=(0.8,0.85))
    bar = plot.Barplot(y=roi_activity, 
                       x='concreteness%composition', 
                       ds=ds, 
                       match='subject', 
                       ylabel='Activity (MNE)',
                       colors=[plt.cm.Blues(0.4),
                               plt.cm.Blues(0.8),
                               plt.cm.Reds(0.4),
                               plt.cm.Reds(0.8)],
                        xticks=['concrete\nword','concrete\nphrase','abstract\nword','abstract\nphrase'],
                        xlabel='Condition',
                        frame='none',
                        w=4,
                        h=4,
                        show=True)
    bar.set_name('ATL activity averaged between 200-250 ms')
    bar.save(op.join(output_dir, 'fig_composition_replicate_bar.png'))
    bar.close()

else:
    for roi in rois:
    
        print(roi)
        ds['stcs'] = stc_reset
        stc_region = stc_reset.sub(source=roi) # subset language network region data
        ds['stcs'] = stc_region # assign this back to the ds
    
        # perform temporal permutation test in a particular region        
        if analysis_type == 'anova':
            output_dir = op.join(results_dir, 'neural/roi/anova', roi, analysis)
            if not op.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            if analysis == 'composition':
                x = 'concreteness*composition*subject'
            elif analysis == 'concreteness':
                x = 'concreteness*subject'
            elif analysis == 'denotation':
                x = 'concreteness*denotation*subject'
            elif analysis == 'specificity':
                x = 'specificity*subject'
                    
            res = testnd.ANOVA(y=ds['stcs'].mean('source'), 
                                x=x, 
                                data=ds, 
                                samples=5000, 
                                pmin=0.05,
                                tstart=0.6,
                                tstop=1.4,
                                match='subject')
            pickle.dump(res, open(op.join(output_dir, f'{roi}.pickle'), 'wb'))
    
            f = open(op.join(output_dir, f'{analysis}_{ch_type}_{roi}_results_table.txt'), 'w')
            f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
            f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s, mintime=??\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
            f.write(str(res.clusters))
            f.close()
        
            pmin = 0.1
            mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
            sign_clusters = res.clusters[mask_sign_clusters]
        
            if sign_clusters.n_cases != None: #check for significant clusters
                for i in range(sign_clusters.n_cases):
                    cluster_nb = i+1
                    cluster = sign_clusters[i]['cluster']
                    tstart = sign_clusters[i]['tstart']
                    tstop = sign_clusters[i]['tstop']
                    effect = sign_clusters[i]['effect']
                    pval = sign_clusters[i]['p']
                    effect = effect.replace(' x ', '%')
        
                    print('Plotting time series for %s' %roi)
                    timecourse = stc_region.mean('source')
                    activation = plot.UTSStat(timecourse, 
                                              effect, 
                                              ds=ds, 
                                              error='sem', 
                                              match='subject', 
                                              legend='lower left', 
                                              xlabel='Time (ms)', 
                                              ylabel='Activity (MNE)', 
                                              xlim=(0.6,1.4), 
                                              title=f'Cluster {i+1}: Effect of {effect} at {roi}, pval={pval}')
                    activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
        
                    activation.save(op.join(output_dir, f'fig_{analysis}_{roi}_cluster{i+1}_timecourse.png'))
                    activation.close()
    
                    ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
                    bar = plot.Barplot(ds['average_source_activation'], 
                                       effect, 
                                       ds=ds, 
                                       title=f'Cluster {i+1}: {round(tstart,3)}-{round(tstop,3)}, {roi}, pval={pval}', 
                                       match='subject', 
                                       ylabel='Activity (MNE)')
                    bar.save(op.join(output_dir, f'fig_{analysis}_{roi}_cluster{i+1}_bar.png'))
                    bar.close()
    # elif analysis_type == 'lm':
    #     output_dir = op.join(results_dir, 'neural/roi/lm', roi, analysis)
    #     if not op.exists(output_dir):
    #         os.makedirs(output_dir, exist_ok=True)
            
    #     if analysis == 'compose':
    #         x = 'concreteness*denotation'
    #     elif analysis == 'specificity':
    #         x = 'specificity'
            
    #     lms = []
    #     for subject in subjects:
            
    #         stcs = []
    #         subject_list = []
    #         condition_list = []

    #         subject = f'sub-{subject}'
    #         print(f'Reading in source estimates from {subject}.')
    #         stc_path = op.join(data_dir, 'stcs', subject)
    #         for condition in conditions:
    #             condition = condition.replace('/','-')
    #             stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}-lh.stc'),
    #                                         subject='fsaverage')
    #             stcs.append(stc)
    #             subject_list.append(subject)
    #             condition_list.append(condition)

    #         ds = Dataset()
            
    #         if analysis == 'compose' or analysis == 'concreteness' or analysis == 'denotation':
    #             condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    #             concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    #             denotation = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
    #         elif analysis == 'specificity':
    #             specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]
            
    #         ds['stcs'] = load.fiff.stc_ndvar(stcs, 
    #                                          subject='fsaverage_src', 
    #                                          src='oct-6', 
    #                                          parc=parc) 
    #         ds['subject'] = Factor(subject_list)
            
    #         if analysis == 'compose' or analysis == 'concreteness' or analysis == 'denotation':    
    #             ds['condition'] = Factor(condition)
    #             ds['condition'].sort_cells(['concrete-baseline',
    #                                         'concrete-subsective',
    #                                         'concrete-privative',
    #                                         'abstract-baseline',
    #                                         'abstract-subsective',
    #                                         'abstract-privative'         
    #                                         ])
    #             ds['concreteness'] = Factor(concreteness)
    #             ds['concreteness'].sort_cells(['concrete','abstract'])
    #             ds['denotation'] = Factor(denotation)
    #             ds['denotation'].sort_cells(['baseline','subsective','privative'])
    #         elif analysis == 'specificity':
    #             ds['specificity'] = Factor(specificity)
    #             ds['specificity'].sort_cells(['low','mid','high'])
    #         # stc_reset = ds['stcs']

    #         print(f'fitting lm to {subject} data.')
    #         lm = testnd.LM(y=ds['stcs'].mean('source'), 
    #                        x='concreteness', 
    #                        data=ds, 
    #                        samples=0, 
    #                        # pmin=0.05,
    #                        tstart=0.6,
    #                        tstop=1.4,
    #                        # tfce=True,
    #                        subject=subject
    #                        )
    #         lms.append(lm)
            
    #     rows = []
    #     for lm in lms:
    #         rows.append([lm.subject, lm.t('intercept'), lm.t('concreteness')])#, lm.t('denotation')])#, lm.t('concreteness x denotation')])
    #     # When creating the dataset for stage 2 analysis, declare subject as random factor;
    #     # this is only relevant if performing ANOVA as stage 2 test.
    #     data = Dataset.from_caselist(['subject', 'intercept', 'concreteness'], rows, random='subject')
    #     data
    #     # pickle.dump(res, open(op.join(output_dir, f'{roi}.pickle'), 'wb'))

            
    #     result = testnd.TTestOneSample('concreteness', data=data, pmin=0.05, tstart=0.6, tstop=1.4)
    #     p = plot.UTS(result)#, t=[0.120, 0.155, None], title=result, head_radius=0.35)
        
    #     timecourse = stc_region.mean('source')
    #     activation = plot.UTSStat(timecourse, 
                                  
    #                               effect, 
    #                               ds=ds, 
    #                               error='sem', 
    #                               match='subject', 
    #                               legend='lower left', 
    #                               xlabel='Time (ms)', 
    #                               ylabel='Activation (MNE)', 
    #                               xlim=(0.6,1.4), 
    #                               title=f'Cluster {i+1}: Effect of {effect} at {roi}')
                
    #     p_cb = p.plot_colorbar(right_of=p.axes[0], label='t')