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
from statsmodels.stats.anova import anova_lm
from mne.stats import permutation_cluster_1samp_test

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config 
import helper
from plot_decoding import *

to_plot = input('denotation+concreteness, denotation_cross_condition, composition: ')
classifier = 'logistic'
data_type = 'MEEG'


subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
figures_dir = op.join(config.project_repo, f'figures/decoding/{to_plot}')
if not op.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

if to_plot == 'denotation+concreteness':
    analyses = ['denotation','concreteness']
elif to_plot == 'composition':
    analyses = ['composition']
elif to_plot == 'denotation_cross_condition':
    analyses = ['subsective','privative']
elif to_plot == 'specificity':
    analyses = ['specificity']
elif to_plot == 'specificity_word':
    analyses = ['specificity_word']
elif to_plot == 'concreteness_word':
    analyses = ['concreteness_word']
    
# read in decoding scores
scores_group = []
if to_plot == 'denotation_cross_condition':
    for analysis in analyses:
        scores = read_decoding_scores(subjects, to_plot, classifier, data_type, evaluation=analysis, timegen=False)
        scores_group.append(scores)
        del scores
else:
    for analysis in analyses:
        scores = read_decoding_scores(subjects, analysis, classifier, data_type, timegen=False)
        scores_group.append(scores)
        del scores

#%% draw fig

if to_plot == 'denotation+concreteness':
    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharey=True, sharex=True)
    gs = GridSpec(2, 1)
elif to_plot == 'denotation_cross_condition':
    fig, axes = plt.subplots(1, 3, figsize=(14,3), sharey=True, sharex=True)
    gs = GridSpec(1, 3, width_ratios=[6,6,2])
else:
    fig, axes = plt.subplots(1, 1, figsize=(10,3))
    gs = GridSpec(1, 1)

for i, analysis in enumerate(analyses):
    
    axis = plt.subplot(gs[i], sharey=fig.get_axes()[0])
        
    color = plt.get_cmap(color_scheme[analysis])(0.7)
    plot_scores(axis, scores_group[i], analysis=analysis, color=color)
    
    # permutation cluster test for sig. effects
    good_clusters = permutation_tests(scores_group[i], timegen=False)
    plot_clusters(good_clusters, scores=scores_group[i], analysis=analysis, ax=axis, color=color)
    
    if (to_plot == 'denotation+concreteness') & (i > 0):
        rect_height = 0.005
        rect_y = 0.54
        rect_x_early = 0.0
        rect_width_early = 0.3
        rect_x_late = 0.6
        rect_width_late = 0.3
        rect = patches.Rectangle((rect_x_early, rect_y), rect_width_early, rect_height,
                             linewidth=0.5, edgecolor='black', facecolor='lightgrey')
        axis.add_patch(rect)
        rect = patches.Rectangle((rect_x_late, rect_y), rect_width_late, rect_height,
                             linewidth=0.5, edgecolor='black', facecolor='lightgrey')
        axis.add_patch(rect)
        
        text_x_early = rect_x_early + rect_width_early / 2
        text_y_early = rect_y + rect_height / 2
        text_x_late = rect_x_late + rect_width_late / 2
        text_y_late = rect_y + rect_height / 2
    
        axis.text(text_x_early, text_y_early, 'adjective', ha='center', va='center', color='black', fontsize=10.5)
        axis.text(text_x_late, text_y_late, 'noun', ha='center', va='center', color='black', fontsize=10.5)
    elif to_plot == 'denotation_cross_condition':
        rect_height = 0.005
        rect_y = 0.48
        rect_x_early = 0.9
        rect_width_early = 0.2
        rect_x_late = 1.1
        rect_width_late = 0.2975
        rect = patches.Rectangle((rect_x_early, rect_y), rect_width_early, rect_height,
                             linewidth=0.5, edgecolor='black', facecolor='lightgrey')
        axis.add_patch(rect)
        rect = patches.Rectangle((rect_x_late, rect_y), rect_width_late, rect_height,
                             linewidth=0.5, edgecolor='black', facecolor='lightgrey')
        axis.add_patch(rect)
        
        text_x_early = rect_x_early + rect_width_early / 2
        text_y_early = rect_y + rect_height / 2
        text_x_late = rect_x_late + rect_width_late / 2
        text_y_late = rect_y + rect_height / 2
    
        axis.text(text_x_early, text_y_early, 'early', ha='center', va='center', color='black', fontsize=10.5)
        axis.text(text_x_late, text_y_late, 'late', ha='center', va='center', color='black', fontsize=10.5)
        
        # test for interaction
        averaged_data = pd.DataFrame(columns=['windows','evaluations','scores'])
        list_windows, list_evaluations, list_scores = [], [], []
        early_window = slice(30,50)
        late_window = slice(50,80)
        for window_name, window in zip(['early','late'],[early_window,late_window]):
            for i, evaluation in enumerate(['subsective','privative']):
                scores_window = scores_group[i][:, window].mean(axis=0)
                list_scores.extend(scores_window)
                list_windows.extend(np.repeat(window_name, len(scores_window)))
                list_evaluations.extend(np.repeat(evaluation, len(scores_window)))
        averaged_data['windows'] = list_windows
        averaged_data['evaluations'] = list_evaluations
        averaged_data['scores'] = list_scores

        model = ols('scores ~ evaluations * windows', data=averaged_data).fit()
        anova_results = anova_lm(model, typ=2)
        print(anova_results)
        anova_results.to_csv(op.join(figures_dir, 'anova_results_window*evaluation.csv'), sep=',')

        # tukey follow up 
        pairwise_scores = []
        k = 0
        for window in ['early','late']:
            for evaluation in ['subsective','privative']:
                print(f'group {k}: {window} {evaluation}')
                group = averaged_data[(averaged_data.windows == window) &
                                      (averaged_data.evaluations == evaluation)].scores.tolist()
                pairwise_scores.append(group)
                k += 1
        posthoc_results = tukey_hsd(pairwise_scores[0],pairwise_scores[1],pairwise_scores[2],pairwise_scores[3])
        print(posthoc_results)
        # posthoc_results.to_csv(op.join(figures_dir, 'followup_pairwise_results_window*evaluation.txt'), sep='\t')
        
        color_palette = [plt.get_cmap(color_scheme['subsective'])(0.7),
                         plt.get_cmap(color_scheme['privative'])(0.7)]
        axis = plt.subplot(gs[2], sharey=fig.get_axes()[0])
        sns.stripplot(
            data=averaged_data,
            x='windows', y='scores', hue='evaluations',
            palette=sns.color_palette(color_palette), 
            dodge=0.7,
            legend=False,
            alpha=0.3,
            ax=axis
        )
        sns.pointplot(
            data=averaged_data,
            x='windows', 
            y="scores", 
            hue='evaluations',
            dodge=.4,
            palette=sns.color_palette(color_palette), 
            errorbar=None,
            markers='o', 
            ax=axis)

#%% adjust appearance

for i, ax in enumerate(fig.get_axes()):
    
    # get rid of spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if to_plot == 'denotation+concreteness':
        
        # set xlim
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(0.48, )
    
        # remove y labels apart from the first subplot
        if i == 0:
            ax.set_xlabel('')
            ax.set_xticks([])
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.spines['bottom'].set_visible(False)
    
        if i == 1:
            ax.set_xticks(np.arange(-0.2, 1.6, 0.2))
    
        # add custom legend
        legend_elements = []
        for analysis in analyses:
            color = plt.get_cmap(color_scheme[analysis])(0.7)
            legend_elements.append(Line2D([0], [0], color=color, lw=3))
        fig.get_axes()[0].legend(legend_elements, ['denotation', 'concreteness'], frameon=False)
    
        axis.set_yticks([0.5, 0.52, 0.54])
        gs.update(wspace=0, hspace=-0.05)

    elif to_plot == 'denotation_cross_condition':

        # remove y labels apart from the first subplot
        if i > 0:
            ax.set_ylabel('')
            plt.setp(ax.get_yticklabels(), visible=False)

        if i < 2:
            # ax.yaxis.tick_right()
            ax.set_xlim(0.6, 1.4)

        # if i == 1:
        #     ax.yaxis.set_ticks_position('both')

        if i == 2:
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel('')
            ax.legend(frameon=False, loc='lower left')
            ax.set_xlabel('Windows')
        
        axis.set_yticks([0.5, 0.52, 0.54])

    else:
        ax.set_xlim(-0.2, 1.4)
        
plt.tight_layout()

#%% save fig 

plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}.png'))
