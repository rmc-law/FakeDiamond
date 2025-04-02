#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:46:57 2024

@author: rl05

Plot decoding time courses of concreteness in subsective vs. privative trials,
and perform statistical tests for early vs. late effects.
"""


import sys
import os
import os.path as op
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import seaborn as sns

from scipy.stats import tukey_hsd
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from mne.stats import permutation_cluster_1samp_test

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config 
from plot_decoding import *

# to_plot = input('denotation+concreteness, concreteness_xcond(_general/_full): ')
to_plot = 'concreteness_xcond'
classifier = 'logistic'
data_type = 'MEEG'
window = 'single'
# window = 'sliding'
if data_type == 'ROI':
    roi = input('roi: ')
else:
    roi = None

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]
print(f'subjects (n={len(subjects)}): ', subjects)
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
figures_dir = op.join(config.project_repo, f'figures/decoding/{to_plot}/{data_type}/{window}')
if not op.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

if to_plot == 'denotation+concreteness':
    analyses = ['denotation','concreteness']
elif to_plot == 'composition':
    analyses = ['composition']
elif to_plot == 'specificity':
    analyses = ['specificity']
elif to_plot == 'specificity_word':
    analyses = ['specificity_word']
elif to_plot == 'concreteness_xcond':
    analyses = ['concreteness_trainWord_testSub','concreteness_trainWord_testPri']
elif to_plot == 'concreteness_xcond_general':
    analyses = ['concreteness_general_testSub','concreteness_general_testPri']
elif to_plot == 'concreteness_xcond_full':
    analyses = ['concreteness_trainSub_testSub','concreteness_trainSub_testPri',
                'concreteness_trainPri_testSub','concreteness_trainPri_testPri']
    
# read in decoding scores
scores_group = []
for analysis in analyses:
    scores = read_decoding_scores(subjects, analysis, classifier, data_type, window=window, roi=roi, timegen=False)
    scores_group.append(scores)
    del scores

if to_plot in ['denotation+concreteness', 'specificity']:
    sfreq = int(scores_group[0].shape[1] / 1.9)
else:
    sfreq = int(scores_group[0].shape[1] / 0.8)
print('sfreq:',sfreq)


#%% set figure style 
FONT = 'Arial'
FONT_SIZE = 15
LINEWIDTH = 1
EDGE_COLOR = 'grey'
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    'axes.labelsize': FONT_SIZE,
    'axes.edgecolor': EDGE_COLOR,
    'axes.linewidth': LINEWIDTH,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'xtick.color': EDGE_COLOR,
    'ytick.color': EDGE_COLOR,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH
})

# %% draw fig

if to_plot == 'denotation+concreteness':
    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharey=True, sharex=True, dpi=300)
    gs = GridSpec(2, 1)
elif to_plot.endswith('_full'):
    fig, axes = plt.subplots(2, 2, figsize=(12,6), sharey=True, sharex=True, dpi=300)
    gs = GridSpec(2, 2, width_ratios=[6,6])
elif to_plot.startswith('concreteness_xcond'):
    # fig, axes = plt.subplots(1, 2, figsize=(12,3), sharey=True, sharex=True, dpi=300)
    # gs = GridSpec(1, 2, width_ratios=[6,6])
    fig, axes = plt.subplots(2, 1, figsize=(6,6), sharey=False, sharex=True, dpi=300)
    gs = GridSpec(2, 1)
elif to_plot.startswith('specificity_word'):
    fig, axes = plt.subplots(1, 1, figsize=(6,3), dpi=300)
    gs = GridSpec(1, 1)
else:
    fig, axes = plt.subplots(1, 1, figsize=(10,3), dpi=300)
    gs = GridSpec(1, 1)

for i, analysis in enumerate(analyses):
    
    axis = plt.subplot(gs[i], sharey=fig.get_axes()[0])
    # axis.set_title(analysis)
        
    # color = plt.get_cmap(color_scheme[analysis])(0.9)
    # plot_scores(axis, scores_group[i], analysis=analysis, color=color, chance=0.5)
    plot_scores(axis, scores_group[i], analysis=analysis, chance=0.5)

    # permutation cluster test for sig. effects
    good_clusters, cluster_pvals = permutation_tests(scores_group[i], timegen=False, against_chance=True)
    print(f'subplot {i+1}')
    # plot_clusters(good_clusters, scores=scores_group[i], analysis=analysis, ax=axis, color=color, cluster_pvals=cluster_pvals)
    plot_clusters(good_clusters, scores=scores_group[i], analysis=analysis, ax=axis, cluster_pvals=cluster_pvals)

    if (to_plot == 'denotation+concreteness') & (i > 0): # this is to add the adjective noun timing indicators
        rect_height = 0.02
        rect_y = 0.56
        rect_x_early = 0.0
        rect_width_early = 0.3
        rect_x_late = 0.6
        rect_width_late = 0.3
        rect = patches.Rectangle((rect_x_early, rect_y), rect_width_early, rect_height,
                             linewidth=0., edgecolor=None, facecolor='lightgrey')
        axis.add_patch(rect)
        rect = patches.Rectangle((rect_x_late, rect_y), rect_width_late, rect_height,
                             linewidth=0., edgecolor=None, facecolor='lightgrey')
        axis.add_patch(rect)
        
        text_x_early = rect_x_early + rect_width_early / 2
        text_y_early = rect_y + rect_height / 2
        text_x_late = rect_x_late + rect_width_late / 2
        text_y_late = rect_y + rect_height / 2

        fontsize = 15
        color_text = 'black'
        axis.text(text_x_early, text_y_early, 'adjective', ha='center', va='center', color=color_text, fontsize=fontsize)
        axis.text(text_x_late, text_y_late, 'noun', ha='center', va='center', color=color_text, fontsize=fontsize)
    elif to_plot.startswith('concreteness_xcond') & (i == 1):

        rect_height = 0.0075
        rect_y = 0.46
        rect_x_early = 0.3
        rect_width_early = 0.2
        rect_x_late = 0.5
        rect_width_late = 0.3
        rect = patches.Rectangle((rect_x_early, rect_y), rect_width_early, rect_height,
                             linewidth=0., edgecolor=None, facecolor='lightgrey')
        axis.add_patch(rect)
        rect = patches.Rectangle((rect_x_late, rect_y), rect_width_late, rect_height,
                             linewidth=0., edgecolor=None, facecolor='grey')
        axis.add_patch(rect)
        
        text_x_early = rect_x_early + rect_width_early / 2
        text_y_early = rect_y + rect_height / 2
        text_x_late = rect_x_late + rect_width_late / 2
        text_y_late = rect_y + rect_height / 2
    
        fontsize = 9
        color_text = 'black'
        axis.text(text_x_early, text_y_early, 'early', ha='center', va='center', color=color_text, fontsize=fontsize)
        axis.text(text_x_late, text_y_late, 'late', ha='center', va='center', color=color_text, fontsize=fontsize)
        
for i, analysis in enumerate(analyses):
    
    axis = plt.subplot(gs[i])

    # get rid of spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)

    if to_plot in ['denotation+concreteness', 'concreteness_xcond']:
        
        # x axis 
        if to_plot == 'denotation+concreteness':
            axis.set_xlim(-0.2, 1.4)
        elif to_plot == 'concreteness_xcond':
            axis.set_xlim(0, 0.8)
            axis.set_xticks([0,0.2,0.4,0.6,0.8])

        # remove x labels apart from the first subplot
        if i == 0:
            axis.set_xlabel('')
            axis.set_xticks([])
            plt.setp(axis.get_xticklabels(), visible=False)
            axis.spines['bottom'].set_visible(False)
        
        # y axis
        if to_plot == 'denotation+concreteness':
            axis.set_yticks([0.5,0.52,0.54, 0.56])
        elif (to_plot == 'concreteness_xcond') & (i == 0):
            axis.set_yticks([0.5,0.52,0.54])
        elif (to_plot == 'concreteness_xcond') & (i == 1):
            axis.set_yticks([0.48,0.5,0.52])
            

        # add vspan to highlight time windows
        if to_plot == 'concreteness_xcond':
            axis.axvspan(0.3, 0.5, alpha=0.3, color='lightgrey') # N4 window
            # axis.axvline(0.3, alpha=0.5, color='lightgrey') # N4 window
            axis.axvspan(0.5, 0.8, alpha=0.65, color='lightgrey') # N/P6 window

        # if i == 1:
        #     axis.set_xticks(np.arange(-0.2, 1.6, 0.2))
    
        # add custom legend
        legend_elements = []
        for analysis in analyses:
            color = plt.get_cmap(color_scheme[analysis])(0.7)
            legend_elements.append(Line2D([0], [0], color=color, lw=3))
        
        if to_plot == 'denotation+concreteness':
            fig.get_axes()[0].legend(legend_elements, ['denotation', 'concreteness'], frameon=False)
        elif to_plot == 'concreteness_xcond':
            fig.get_axes()[0].legend(legend_elements, ['test on subsective', 'test on privative'], 
                                     loc='upper left', frameon=False)

        # axis.set_yticks([0.5, 0.54, 0.58])
        gs.update(wspace=0., hspace=0.)

    # elif to_plot.startswith('concreteness_xcond'):

    #     if to_plot.endswith('full') and i in [0, 1]:
    #         axis.spines['bottom'].set_visible(False)
    #         axis.set_xlabel('')
    #         plt.setp(axis.get_xticklabels(), visible=False)
    #         axis.tick_params(bottom=False)
    #         axis.set_yticks([0.48, 0.5, 0.52, 0.54])
    #     elif to_plot.endswith('general'):
    #         axis.set_yticks([0.48, 0.5, 0.52, 0.54])
        
    #     if i in [1,3]:
            
    #         axis.set_ylabel('')
    #         plt.setp(axis.get_yticklabels(), visible=False)

    #         # axis.set_ylim(0.47, 0.53)

    #     axis.set_xticks(np.arange(0., 1.0, 0.2))
    #     axis.set_xlim(0., 0.8)
    elif to_plot.startswith('specificity_word'):
        axis.set_xlim(0., 0.8)
    else:
        axis.set_xlim(-0.2, 1.4)
        
plt.tight_layout()

if roi:
    plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_{roi}_{sfreq}Hz.png'))
else:
    plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_{sfreq}Hz.png'))

plt.close()
    
#%% plot interaction in cross condition decoding

if to_plot.startswith('concreteness_x'):
    averaged_data = pd.DataFrame(columns=['timewindow','evaluation','score','subject','train_on'])
    list_timewindows, list_evaluations, list_scores, list_train_on = [], [], [], []
    if window == 'single':
        if sfreq == 100:
            timewindow_early = dict(start=30,stop=50)
            timewindow_late = dict(start=50,stop=80)
        elif sfreq == 250:
            timewindow_early = slice(75,125)
            timewindow_late = slice(125,200)
    elif window == 'sliding':
        if sfreq == 47:
            timewindow_early = slice(14,24)
            timewindow_late = slice(24,38)
        elif sfreq == 50:
            timewindow_early = slice(15,25)
            timewindow_late = slice(25,40)
        elif sfreq == 95:
            timewindow_early = dict(start=29,stop=48)
            timewindow_late = dict(start=48,stop=76)
        elif sfreq == 97:
            timewindow_early = dict(start=29,stop=49)
            timewindow_late = dict(start=49,stop=78)
        elif sfreq == 118:
            timewindow_early = dict(start=36,stop=60)
            timewindow_late = dict(start=60,stop=95)

    if to_plot == 'concreteness_xcond_full':
        for i, (evaluation, train_on) in enumerate(zip(['subsective','privative','subsective','privative'],['subsective','subsective','privative','privative'])):
            for timewindow_name, timewindow in zip(['early','late'],[timewindow_early,timewindow_late]):
                scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
                list_scores.extend(scores_timewindow)
                list_timewindows.extend(np.repeat(timewindow_name, len(scores_timewindow)))
                list_evaluations.extend(np.repeat(evaluation, len(scores_timewindow)))
                list_train_on.extend(np.repeat(train_on, len(scores_timewindow)))
        averaged_data['timewindow'] = list_timewindows
        averaged_data['evaluation'] = list_evaluations
        averaged_data['score'] = list_scores
        averaged_data['subject'] = subjects * 8
        averaged_data['train_on'] = list_train_on
    else:
        for i, evaluation in enumerate(['subsective','privative']):
            for timewindow_name, timewindow in zip(['early','late'],[timewindow_early,timewindow_late]):
                scores_timewindow = scores_group[i][:, timewindow['start']:timewindow['stop']].mean(axis=1)
                list_scores.extend(scores_timewindow)
                list_timewindows.extend(np.repeat(timewindow_name, len(scores_timewindow)))
                list_evaluations.extend(np.repeat(evaluation, len(scores_timewindow)))
        averaged_data['timewindow'] = list_timewindows
        averaged_data['evaluation'] = list_evaluations
        averaged_data['score'] = list_scores
        averaged_data['subject'] = subjects * 4
    
       
    color_palette = [plt.get_cmap(color_scheme['concreteness_trainWord_testSub'])(0.7),
                     plt.get_cmap(color_scheme['concreteness_trainWord_testPri'])(0.7)]
    
    if to_plot == 'concreteness_xcond_full':
        for i, train_on in enumerate(['subsective','privative']):
            filtered_averaged_data = averaged_data[averaged_data.train_on == train_on]
            fig, axis = plt.subplots(figsize=(2.75,3)) 
            sns.barplot(
                data=filtered_averaged_data,
                x='timewindow', y='score', hue='evaluation',
                palette=sns.color_palette(color_palette), 
                # alpha=0.3,
                errorbar='se',
                ax=axis
            )
            axis.set(ylim=(0.45,0.55))
            axis.set(xlabel='time window', ylabel='')
            plt.gca().legend().set_title('')
            plt.tight_layout()
    
            if roi:
                plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_{roi}_{sfreq}Hz_bar.png'))
            else:
                plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_train-on-{train_on}_{sfreq}Hz_bar.png'))

    else:
        fig, axis = plt.subplots(figsize=(2.75,3.15)) 
        sns.barplot(
            data=averaged_data,
            x='timewindow', y='score', hue='evaluation',
            palette=sns.color_palette(color_palette), 
            # alpha=0.3,
            errorbar='se',
            ax=axis
        )
        axis.set(ylim=(0.45,0.55), yticks=[0.45,0.50,0.55])
        axis.set(xlabel='time window', ylabel='AUC')
        plt.gca().legend(frameon=False).set_title('') # , loc='upper right', bbox_to_anchor=(1., 1.35)

        plt.tight_layout()

        if roi:
            plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_{roi}_{sfreq}Hz_bar.png'))
        else:
            plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_{sfreq}Hz_bar.png'))

#%% save fig 

if roi:
    plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}_{roi}.png'))
else:
    plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}_{window}.png'))
