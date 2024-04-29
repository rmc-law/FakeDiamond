#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:46:46 2024

@author: rl05
"""

import os
import os.path as op
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


from eelbrain import Dataset, load, Factor, test, plot
from eelbrain._stats.stats import variability
from mne import (read_epochs, read_source_spaces, read_labels_from_annot,
                 read_source_estimate)

import config 
from helper import calculate_avg_sem

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
print(f'subjects (n={len(subjects)}): ', subjects)
data_dir = op.join(config.project_repo, 'data')
figs_dir = op.join(config.project_repo, 'figures')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
fsaverage_src_fname = op.join(subjects_dir, 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src_fsaverage = read_source_spaces(fsaverage_src_fname, verbose=False)
stc_path = op.join(data_dir, 'stcs')
analysis = input('analysis? (replicate, composition): ')

# label annot
parc = 'fake_diamond'
annot = read_labels_from_annot('fsaverage_src', parc=parc, hemi='lh')[:-1] # + read_labels_from_annot('fsaverage_src',parc='ventral_ATL')[:2]
del annot[2] # to delete visual sources
assert len(annot) == 4 # should be four rois

results_dir = '/imaging/hauk/rl05/fake_diamond/results/neural/roi/anova/'
times = np.linspace(0., 0.8, 200)

#%% set figure style 
FONT = 'Arial'
FONT_SIZE = 15
LINEWIDTH = 1.5
EDGE_COLOR = 'grey'
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    'axes.labelsize': FONT_SIZE,
    'axes.edgecolor': EDGE_COLOR,
    'axes.linewidth': LINEWIDTH,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'xtick.color': EDGE_COLOR,
    'ytick.color': EDGE_COLOR,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.major.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH
})

#%% read in dataset using eelbrain

if analysis in ['composition','replicate']:
    conditions = ['concrete-baseline','concrete-subsective','abstract-baseline','abstract-subsective']
    roi_to_plot = input('ATL or all: ')
    colors = plt.cm.Greys([0.4, 0.8])
    colors_bar = [plt.cm.Blues(0.4),plt.cm.Blues(0.8),plt.cm.Reds(0.4),plt.cm.Reds(0.8)]
    relevant_conditions = ['baseline','subsective']
if analysis == 'denotation': 
    conditions = ['concrete-subsective','concrete-privative','abstract-subsective','abstract-privative']
    roi_to_plot = 'ATL'
    colors = plt.cm.YlGn([0.4, 0.8])
    colors_bar = 
    relevant_conditions = ['subsective','privative']

subjects_list, conditions_list, stcs = [], [], []
for subject in subjects:
    print(f'Reading in stc {subject}.')
    for condition in conditions:
        stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_MEEG-lh.stc')
        stc = read_source_estimate(stc_fname, subject='fsaverage_src')
        stc = stc.crop(tmin=0.6, tmax=1.4)
        stc.tmin = 0.
        stcs.append(stc)
        subjects_list.append(subject)
        conditions_list.append(condition)
        del stc

ds = Dataset()
concreteness = [condition.split('-')[0] for condition in conditions_list]
if analysis in ['composition','replicate']:
    composition = [condition.split('-')[1] for condition in conditions_list]
elif analysis == 'denotation': 
    denotation = [condition.split('-')[1] for condition in conditions_list]
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage_src', src='oct-6', parc=parc) 
ds['subject'] = Factor(subjects_list, random=True)
ds['condition'] = Factor(conditions_list)
ds['condition'].sort_cells(conditions)
ds['concreteness'] = Factor(concreteness)
ds['concreteness'].sort_cells(['concrete','abstract'])
if analysis in ['composition','replicate']:
    ds['composition'] = Factor(composition)
    ds['composition'].sort_cells(['baseline','subsective'])
elif analysis == 'denotation': 
    ds['denotation'] = Factor(denotation)
    ds['denotation'].sort_cells(['subsective','privative'])
stc_reset = ds['stcs']


#%% plot figures
    
fig_dir = op.join(config.project_repo, 'figures', 'univariate')

if analysis == 'replicate':
    roi = 'anteriortemporal-lh'
    ds['stcs'] = stc_reset
    stc_region = stc_reset.sub(source=roi) # subset language network region data
    ds['stcs'] = stc_region # assign this back to the ds
    output_dir = op.join(results_dir, 'neural/roi/anova', roi, 'replicateB&P')
    if not op.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    x = 'concreteness*composition*subject'
    result = test.ANOVA(y=ds['stcs'].mean('source').mean(time=(0.2,0.25)), x=x, data=ds)
    print(result, file=open(op.join(output_dir, 'anova.txt'), 'w'))
    
    roi_activity = stc_region.mean('source').mean(time=(0.2,0.25))
    bar = plot.Barplot(y=roi_activity, 
                       x='concreteness%composition', 
                       ds=ds, 
                       match='subject', 
                       colors=colors_bar,
                        xticks=['concrete\nword','concrete\nphrase','abstract\nword','abstract\nphrase'],
                        xlabel='Condition',
                        ylabel='ATL activity (MNE)',
                        frame='none',
                        w=5,
                        h=4,
                        show=True)
    bar.set_name('ATL activity averaged between 200-250 ms')
    bar.save(op.join(output_dir, 'fig_composition_replicateB&P2011_bars.png'))
    bar.save(op.join(fig_dir, 'fig_composition_replicateB&P2011_bars.png'))
    bar.close()

else:
    if roi_to_plot == 'ATL':
        rois = ['anteriortemporal-lh']
        roi_names = ['anterior temporal']
        n_subplots = len(rois)
        fig_name = op.join(fig_dir, f'fig_{analysis}_ATL.png')
    else:
        rois = ['anteriortemporal-lh', 'posteriortemporal-lh', 'inferiorfrontal-lh', 'temporoparietal-lh']
        roi_names = ['anterior temporal', 'posterior temporal', 'inferior frontal', 'temporo parietal']
        n_subplots = len(rois)
        fig_name = op.join(fig_dir, f'fig_{analysis}_allROIs.png')

    scaling_factor = 4
    fig_height = scaling_factor * n_subplots
    if n_subplots == 1:
        fig, axis = plt.subplots(n_subplots, sharex=True, sharey=False, dpi=300)
    else:
        fig, axes = plt.subplots(n_subplots, sharex=True, sharey=False, dpi=300)
    fig.set_size_inches(8, fig_height)
    # fig.subplots_adjust(hspace=0.45)

    for i_roi, roi in enumerate(rois):
        if roi_to_plot == 'all':
            axis = axes[i_roi]
        ds['stcs'] = stc_reset
        stcs = ds['stcs']
        stcs_region = stcs.sub(source = roi)
        time_courses = stcs_region.mean('source')
        ds['stcs'] = time_courses
        for relevant_condition, color in zip(relevant_conditions,colors):
            data = ds['stcs'][ds[analysis].isin([relevant_condition])]
            data_group_avg = data.mean('case') # average over subjects
            error = variability(y=data.x[:,:], x=ds['condition'], match=ds['subject'], pool=True, spec='sem')
            axis.plot(times, data_group_avg.x, color=color, lw=2.5)
            axis.fill_between(times, data_group_avg.x-error, data_group_avg.x+error, alpha=0.1, color=color)
        axis.set_xlim(0., 0.8)

        # add custom legend
        legend_elements = [Line2D([0], [0], color=color, lw=3) for color in colors]
        if analysis in ['composition','replicate']:
            fig.get_axes()[0].legend(legend_elements, ['word', 'phrases'], loc='upper right')
        elif analysis == 'denotation': 
            fig.get_axes()[0].legend(legend_elements, ['subsective', 'privative'], loc='upper right')

        # read in permutation test pickle file
        print('unloading pickle: ', roi)
        pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
        with open(pickle_fname, 'rb') as f:
            res = pickle.load(f)
        pmin = 0.1
        mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
        sign_clusters = res.clusters[mask_sign_clusters]

        if sign_clusters.n_cases != None: #check for significant clusters
            for i in range(sign_clusters.n_cases):
                cluster_tstart = sign_clusters[i]['tstart'] - 0.6
                cluster_tstop = sign_clusters[i]['tstop'] - 0.6
                cluster_effect = sign_clusters[i]['effect']
                cluster_pval = sign_clusters[i]['p']
                if cluster_effect != analysis:
                    continue
                if cluster_pval < 0.05:
                    alpha = 0.3
                else:
                    alpha = 0.15
                axis.axvspan(cluster_tstart, cluster_tstop, color='yellow', alpha=alpha, zorder=-50)

    xticks = [0., 0.2, 0.4, 0.6, 0.8]
    plt.xticks(xticks)
    plt.xlabel('Time (s)')
    if roi_to_plot == 'ATL':
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(True)
    if analysis == 'denotation':
        plt.ylabel('Activity (MNE)')
    plt.tight_layout()
    fig.savefig(fig_name)
    plt.close()


#%% old
fig.set_size_inches(12, 5)
fig.subplots_adjust(hspace=0.45)    
    
label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
label.subject = 'fsaverage_src'

group_average = np.zeros((len(subjects), len(conditions), len(times)))

for i_subject, subject in enumerate(subjects): 

    subject = f'sub-{subject}'
        
    for j_condition, condition in enumerate(conditions):

        condition = condition.replace('/','-')
        stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
        stc = read_source_estimate(stc_fname)
        time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
        group_average[i_subject, j_condition, :] = time_course

group_average_phrase = np.array([group_average[:,0],group_average[:,3]]).mean(axis=0)        
group_average_word = np.array([group_average[:,2],group_average[:,5]]).mean(axis=0)

# subset only word 2 window
target_window = (0., 0.8)
target_window_start_index = int((target_window[0] - times[0]) * 250)
target_window_stop_index = int((target_window[1] - times[0]) * 250)
group_average_phrase = group_average_phrase[:, target_window_start_index:target_window_stop_index]
group_average_word = group_average_word[:, target_window_start_index:target_window_stop_index]
target_window_times = stc.times[target_window_start_index:target_window_stop_index] - 0.6 # make 0 the onset of noun

avg_phrase, sem_phrase = calculate_avg_sem(group_average_phrase)
avg_word, sem_word = calculate_avg_sem(group_average_word)

avgs = [avg_phrase, avg_word]
sems = [sem_phrase, sem_word]


# group_average_phrase_concrete = group_average[:,0]
# group_average_phrase_abstract = group_average[:,3]
# group_average_word_concrete = group_average[:,2]
# group_average_word_abstract = group_average[:,5]

# # subset only word 2 window
# target_window = (0., 0.8)
# target_window_start_index = int((target_window[0] - times[0]) * 250)
# target_window_stop_index = int((target_window[1] - times[0]) * 250)
# group_average_phrase_concrete = group_average_phrase_concrete[:, target_window_start_index:target_window_stop_index]
# group_average_phrase_abstract = group_average_phrase_abstract[:, target_window_start_index:target_window_stop_index]
# group_average_word_concrete = group_average_word_concrete[:, target_window_start_index:target_window_stop_index]
# group_average_word_abstract = group_average_word_abstract[:, target_window_start_index:target_window_stop_index]
# target_window_times = stc.times[target_window_start_index:target_window_stop_index] - 0.6 # make 0 the onset of noun

# avg_phrase_concrete, sem_phrase_concrete = calculate_avg_sem(group_average_phrase_concrete)
# avg_phrase_abstract, sem_phrase_abstract = calculate_avg_sem(group_average_phrase_abstract)
# avg_word_concrete, sem_word_concrete = calculate_avg_sem(group_average_word_concrete)
# avg_word_abstract, sem_word_abstract = calculate_avg_sem(group_average_word_abstract)

# avgs = [avg_word_concrete, avg_phrase_concrete, avg_word_abstract, avg_phrase_abstract]
# sems = [sem_word_concrete, sem_phrase_concrete, sem_word_abstract, sem_phrase_abstract]


for avg, sem, color in zip(avgs, sems, colors):
    
    axis.plot(target_window_times, avg, color=color, alpha=0.9)
    axis.fill_between(target_window_times, avg-sem, avg+sem, color=color, alpha=0.1)

axis.set_title(name)
plt.xlabel('Time (s)')
# plt.ylabel('Activity (MNE)')

# read in permutation test pickle file
print('unloading pickle: ', roi)
pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
with open(pickle_fname, 'rb') as f:
    res = pickle.load(f)
pmin = 0.1
mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
sign_clusters = res.clusters[mask_sign_clusters]

if sign_clusters.n_cases != None: #check for significant clusters
    for i in range(sign_clusters.n_cases):
        cluster_tstart = sign_clusters[i]['tstart'] - 0.6
        cluster_tstop = sign_clusters[i]['tstop'] - 0.6
        cluster_effect = sign_clusters[i]['effect']
        cluster_pval = sign_clusters[i]['p']
        if cluster_effect != analysis:
            continue
        if cluster_pval < 0.05:
            alpha = 0.3
        else:
            alpha = 0.15
        axis.axvspan(cluster_tstart, cluster_tstop, color='yellow', alpha=alpha, zorder=-50)

fig_dir = op.join(config.project_repo, 'figures', 'group')
fig.savefig(op.join(fig_dir, f'fig_{analysis}_{roi}.png'))
plt.close()


#%% main effect of composition in all four ROIs
    
analysis = 'composition'

conditions_compose = conditions[:6]
colors = plt.cm.Greys([0.8, 0.4])
        
number_of_subplots = len(annot)
scaling_factor = 3
fig_height = scaling_factor * number_of_subplots
fig, axes = plt.subplots(number_of_subplots, sharex=True, sharey=False)
fig.set_size_inches(10, fig_height)
fig.subplots_adjust(hspace=0.45)
# fig.suptitle(f'ROI activity averaged over subjects (n={len(subjects)})')
    

for roi, name, axis in zip(rois, roi_names, axes.ravel()):
    
    label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
    label.subject = 'fsaverage_src'

    group_average = np.zeros((len(subjects), len(conditions_compose), len(times)))

    for i_subject, subject in enumerate(subjects): 
    
       subject = f'sub-{subject}'
           
       for j_condition, condition in enumerate(conditions_compose):

           condition = condition.replace('/','-')
           stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
           stc = read_source_estimate(stc_fname)
           time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
           group_average[i_subject, j_condition, :] = time_course
   
    group_average_phrase = np.array([group_average[:,0],group_average[:,3]]).mean(axis=0)        
    group_average_word = np.array([group_average[:,2],group_average[:,5]]).mean(axis=0)
    
    # subset only word 2 window
    target_window = (0., 0.8)
    target_window_start_index = int((target_window[0] - times[0]) * 250)
    target_window_stop_index = int((target_window[1] - times[0]) * 250)
    group_average_phrase = group_average_phrase[:, target_window_start_index:target_window_stop_index]
    group_average_word = group_average_word[:, target_window_start_index:target_window_stop_index]
    target_window_times = stc.times[target_window_start_index:target_window_stop_index] - 0.6 # make 0 the onset of noun

    avg_phrase, sem_phrase = calculate_avg_sem(group_average_phrase)
    avg_word, sem_word = calculate_avg_sem(group_average_word)

    avgs = [avg_phrase, avg_word]
    sems = [sem_phrase, sem_word]
    
    xticks = [0., 0.2, 0.4, 0.6, 0.8]
    plt.xticks(xticks)

    for avg, sem, color in zip(avgs, sems, colors):
        
        axis.plot(target_window_times, avg, color=color, alpha=0.9)
        axis.fill_between(target_window_times, avg-sem, avg+sem, color=color, alpha=0.1)

    axis.set_title(name)
    axis.set_xlim(0., 0.8)
    plt.xlabel('Time (s)')
    # plt.ylabel('Activity (MNE)')

    # read in permutation test pickle file
    print('unloading pickle: ', roi)
    pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
    with open(pickle_fname, 'rb') as f:
        res = pickle.load(f)
    pmin = 0.1
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_tstart = sign_clusters[i]['tstart'] - 0.6
            cluster_tstop = sign_clusters[i]['tstop'] - 0.6
            cluster_effect = sign_clusters[i]['effect']
            cluster_pval = sign_clusters[i]['p']
            if cluster_effect != analysis:
                continue
            if cluster_pval < 0.05:
                alpha = 0.3
            else:
                alpha = 0.15
            axis.axvspan(cluster_tstart, cluster_tstop, color='yellow', alpha=alpha, zorder=-50)

fig_dir = op.join(config.project_repo, 'figures', 'group')
fig.savefig(op.join(fig_dir, f'fig_{analysis}_clusters.png'))
plt.close()
    

#%% main effect of denotation
    
analysis = 'denotation'

conditions_compose = conditions[:6]
times = np.linspace(-1.1, 0.8, 475)
colors = [plt.cm.Blues(0.5), plt.cm.Reds(0.5)]
        
number_of_subplots = len(annot)
scaling_factor = 3
fig_height = scaling_factor * number_of_subplots
fig, axes = plt.subplots(number_of_subplots, sharex=True, sharey=False)
fig.set_size_inches(10, fig_height)
fig.subplots_adjust(hspace=0.45)
# fig.suptitle(f'ROI activity averaged over subjects (n={len(subjects)})')
    

for roi, name, axis in zip(rois, roi_names, axes.ravel()):
    
    label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
    label.subject = 'fsaverage_src'

    group_average = np.zeros((len(subjects), len(conditions_compose), len(times)))

    for i_subject, subject in enumerate(subjects): 
    
       subject = f'sub-{subject}'
           
       for j_condition, condition in enumerate(conditions_compose):

           condition = condition.replace('/','-')
           stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
           stc = read_source_estimate(stc_fname)
           time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
           group_average[i_subject, j_condition, :] = time_course

    group_average_subsective = np.array([group_average[:,0],group_average[:,3]]).mean(axis=0)   
    group_average_privative = np.array([group_average[:,1],group_average[:,4]]).mean(axis=0)        
    
    # subset only word 2 window
    target_window = (0., 0.8)
    target_window_start_index = int((target_window[0] - times[0]) * 250)
    target_window_stop_index = int((target_window[1] - times[0]) * 250)
    group_average_subsective = group_average_subsective[:, target_window_start_index:target_window_stop_index]
    group_average_privative = group_average_privative[:, target_window_start_index:target_window_stop_index]
    target_window_times = stc.times[target_window_start_index:target_window_stop_index] - 0.6 # make 0 the onset of noun

    avg_subsective, sem_subsective = calculate_avg_sem(group_average_subsective)
    avg_privative, sem_privative = calculate_avg_sem(group_average_privative)

    avgs = [avg_subsective, avg_privative]
    sems = [sem_subsective, sem_privative]
    
    xticks = [0., 0.2, 0.4, 0.6, 0.8]
    plt.xticks(xticks)

    for avg, sem, color in zip(avgs, sems, colors):
        
        axis.plot(target_window_times, avg, color=color, alpha=0.9)
        axis.fill_between(target_window_times, avg-sem, avg+sem, color=color, alpha=0.1)

    axis.set_title(name)
    axis.set_xlim(0., 0.8)
    plt.xlabel('Time (s)')
    # plt.ylabel('Activity (MNE)')

    # read in permutation test pickle file
    print('unloading pickle: ', roi)
    pickle_fname = op.join(results_dir, f'{roi}/{analysis}/{roi}.pickle')
    with open(pickle_fname, 'rb') as f:
        res = pickle.load(f)
    pmin = 0.1
    mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
    sign_clusters = res.clusters[mask_sign_clusters]

    if sign_clusters.n_cases != None: #check for significant clusters
        for i in range(sign_clusters.n_cases):
            cluster_tstart = sign_clusters[i]['tstart'] - 0.6
            cluster_tstop = sign_clusters[i]['tstop'] - 0.6
            cluster_effect = sign_clusters[i]['effect']
            cluster_pval = sign_clusters[i]['p']
            if cluster_effect != analysis:
                continue
            if cluster_pval < 0.05:
                alpha = 0.3
            else:
                alpha = 0.15
            axis.axvspan(cluster_tstart, cluster_tstop, color='yellow', alpha=alpha, zorder=-50)

fig_dir = op.join(config.project_repo, 'figures', 'group')
fig.savefig(op.join(fig_dir, f'fig_{analysis}_clusters.png'))
plt.close()


# # =============================================================================
# # analysing concreteness main effect
# # =============================================================================

# # apply the custom style
# plt.rcParams.update(poster_style)

# annot = read_labels_from_annot('fsaverage_src', parc='fake_diamond', hemi='lh')[:-1] # + read_labels_from_annot('fsaverage_src',parc='ventral_ATL')[:2]
# del annot[2] # to delete visual sources
# assert len(annot) == 4 # should be four rois
# rois = ['anteriortemporal-lh', 'posteriortemporal-lh',
#         'inferiorfrontal-lh', 'temporoparietal-lh']
# roi_names = ['anterior temporal', 'posterior temporal',
#              'inferior frontal', 'temporo parietal']

# conditions_compose = [conditions[2],conditions[5]]
# times = np.linspace(-0.5, 1.4, 475)
# colors = np.array([plt.cm.Oranges(0.8),plt.cm.Purples(0.8)])
        
# number_of_subplots = len(annot)
# scaling_factor = 3
# fig_height = scaling_factor * number_of_subplots
# fig, axes = plt.subplots(number_of_subplots, sharex=True, sharey=False)
# fig.set_size_inches(10, fig_height)
# fig.subplots_adjust(hspace=0.45)
# # fig.suptitle(f'ROI activity averaged over subjects (n={len(subjects)})')
    

# for roi, name, axis in zip(rois, roi_names, axes.ravel()):
    
#     label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
#     label.subject = 'fsaverage_src'

#     group_average = np.zeros((len(subjects), len(conditions_compose), len(times)))

#     for i_subject, subject in enumerate(subjects): 
    
#        subject = f'sub-{subject}'
           
#        for j_condition, condition in enumerate(conditions_compose):
                      
#            condition = condition.replace('/','-')
#            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
#            stc = read_source_estimate(stc_fname)
           
#            # apply baseline here because MNE is not noise normalised, and condition SNRs are different
#            stc = stc.apply_baseline(baseline=(None,-0.3)) 
           
#            time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
           
#            group_average[i_subject, j_condition, :] = time_course
   
#     avg_compose, sem_compose = calculate_avg_sem(group_average[:,0])
#     avg_baseline, sem_baseline = calculate_avg_sem(group_average[:,1])


#     avgs = [avg_compose, avg_baseline]
#     sems = [sem_compose, sem_baseline]
    
#     xticks = [-0.3, 0, 0.6, 1.2]
#     plt.xticks(xticks)
#     for xtick in xticks[:3]:
#         axis.axvline(x=xtick, color='lightgray', lw=3)
        
#     for avg, sem, color in zip(avgs, sems, colors):
        
#         axis.plot(stc.times, avg, color=color, alpha=0.9)
#         axis.fill_between(stc.times, avg-sem, avg+sem, color=color, alpha=0.1)

#     axis.set_title(name)
#     plt.xlabel('Time (s)')

# fig_dir = op.join(config.project_repo, 'figures', 'group')
# fig.savefig(op.join(fig_dir, f'concreteness_{ch_type}_group(n={len(subjects)}).png'))
# plt.close()
    


    
# # =============================================================================
# # denotation main effect
# # =============================================================================

# # apply the custom style
# plt.rcParams.update(poster_style)

# annot = read_labels_from_annot('fsaverage_src', parc='fake_diamond', hemi='lh')[:-1] # + read_labels_from_annot('fsaverage_src',parc='ventral_ATL')[:2]
# del annot[2] # to delete visual sources
# assert len(annot) == 4 # should be four rois
# rois = ['anteriortemporal-lh', 'posteriortemporal-lh',
#         'inferiorfrontal-lh', 'temporoparietal-lh']
# roi_names = ['anterior temporal lobe', 'posterior temporal lobe',
#              'inferior frontal cortex', 'temporo parietal junction']

# conditions_compose = conditions[:6]
# times = np.linspace(-0.5, 1.4, 475)
# colors = [plt.cm.Greys(0.5), plt.cm.Blues(0.5), plt.cm.Reds(0.5)]
        
# number_of_subplots = len(annot)
# scaling_factor = 3
# fig_height = scaling_factor * number_of_subplots
# fig, axes = plt.subplots(number_of_subplots, sharex=True, sharey=False)
# fig.set_size_inches(10, fig_height)
# fig.subplots_adjust(hspace=0.45)
# # fig.suptitle(f'ROI activity averaged over subjects (n={len(subjects)})')
    

# for roi, name, axis in zip(rois, roi_names, axes.ravel()):
    
#     label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
#     label.subject = 'fsaverage_src'

#     group_average = np.zeros((len(subjects), len(conditions_compose), len(times)))

#     for i_subject, subject in enumerate(subjects): 
    
#        subject = f'sub-{subject}'
           
#        for j_condition, condition in enumerate(conditions_compose):
                      
#            condition = condition.replace('/','-')
#            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
#            stc = read_source_estimate(stc_fname)
           
#            # apply baseline here because MNE is not noise normalised, and condition SNRs are different
#            stc = stc.apply_baseline(baseline=(None,-0.3)) 
           
#            time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
           
#            group_average[i_subject, j_condition, :] = time_course
   
#     group_average_subsective = np.array([group_average[:,0],
#                                       group_average[:,3]
#                                       ]).mean(axis=0)   
#     group_average_privative = np.array([group_average[:,1],
#                                       group_average[:,4]
#                                       ]).mean(axis=0)        
#     group_average_baseline = np.array([group_average[:,2],
#                                       group_average[:,5]
#                                       ]).mean(axis=0)    
    
#     avg_subsective, sem_subsective = calculate_avg_sem(group_average_subsective)
#     avg_privative, sem_privative = calculate_avg_sem(group_average_privative)
#     avg_baseline, sem_baseline = calculate_avg_sem(group_average_baseline)


#     avgs = [avg_baseline, avg_subsective, avg_privative]
#     sems = [sem_baseline, sem_subsective, sem_privative]
    
#     xticks = [-0.3, 0, 0.6, 1.2]
#     plt.xticks(xticks)
#     for xtick in xticks[:3]:
#         axis.axvline(x=xtick, color='lightgray', lw=3)
        
#     for avg, sem, color in zip(avgs, sems, colors):
        
#         axis.plot(stc.times, avg, color=color, alpha=0.9)
#         axis.fill_between(stc.times, avg-sem, avg+sem, color=color, alpha=0.1)

#     axis.set_title(name)
#     plt.xlabel('Time (s)')

# fig_dir = op.join(config.project_repo, 'figures', 'group')
# fig.savefig(op.join(fig_dir, f'denotation_{ch_type}_group(n={len(subjects)}).png'))
# plt.close()






# # =============================================================================
# # analysing specificity main effect
# # =============================================================================

# annot = read_labels_from_annot('fsaverage_src', parc='fake_diamond', hemi='lh')[:-1] # + read_labels_from_annot('fsaverage_src',parc='ventral_ATL')[:2]
# del annot[2] # to delete visual sources
# assert len(annot) == 4 # should be four rois
# rois = ['anteriortemporal-lh', 'posteriortemporal-lh',
#         'inferiorfrontal-lh', 'temporoparietal-lh']
# roi_names = ['anterior temporal', 'posterior temporal',
#              'inferior frontal', 'temporo parietal']

# conditions_specificity = conditions[6:]
# times = np.linspace(-0.5, 1.4, 475)
# colors = plt.cm.Greens([0.3,0.6,0.8])
        
# number_of_subplots = len(annot)
# scaling_factor = 3
# fig_height = scaling_factor * number_of_subplots
# fig, axes = plt.subplots(number_of_subplots, sharex=True, sharey=False)
# fig.set_size_inches(10, fig_height)
# fig.subplots_adjust(hspace=0.45)
# # fig.suptitle(f'ROI activity averaged over subjects (n={len(subjects)})')
    

# for roi, name, axis in zip(rois, roi_names, axes.ravel()):
    
#     label = [label for label in annot if label.name == roi][0] # use this to order rois for plotting
#     label.subject = 'fsaverage_src'

#     group_average = np.zeros((len(subjects), len(conditions_specificity), len(times)))

#     for i_subject, subject in enumerate(subjects): 
    
#        subject = f'sub-{subject}'
           
#        for j_condition, condition in enumerate(conditions_specificity):
                      
#            condition = condition.replace('/','-')
#            stc_fname = op.join(stc_path, subject, f'{subject}_{condition}_{ch_type}-lh.stc')
#            stc = read_source_estimate(stc_fname)
           
#            # apply baseline here because MNE is not noise normalised, and condition SNRs are different
#            stc = stc.apply_baseline(baseline=(None,-0.3)) 
           
#            time_course = stc.extract_label_time_course(label, src_fsaverage, mode='mean')[0]
           
#            group_average[i_subject, j_condition, :] = time_course
   
#     avg_low, sem_low = calculate_avg_sem(group_average[:,0])
#     avg_mid, sem_mid = calculate_avg_sem(group_average[:,1])
#     avg_high, sem_high = calculate_avg_sem(group_average[:,2])


#     avgs = [avg_low, avg_mid, avg_high]
#     sems = [sem_low, sem_mid, sem_high]
    
#     xticks = [-0.3, 0, 0.6, 1.2]
#     plt.xticks(xticks)
#     for xtick in xticks[:3]:
#         axis.axvline(x=xtick, color='lightgray', lw=3)
        
#     for avg, sem, color in zip(avgs, sems, colors):
        
#         axis.plot(stc.times, avg, color=color, alpha=0.9)
#         axis.fill_between(stc.times, avg-sem, avg+sem, color=color, alpha=0.1)

#     axis.set_title(name)
#     plt.xlabel('Time (s)')

# fig_dir = op.join(config.project_repo, 'figures', 'group')
# fig.savefig(op.join(fig_dir, f'specificity_{ch_type}_group(n={len(subjects)}).png'))
# plt.close()

    
