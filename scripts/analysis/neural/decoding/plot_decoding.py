#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:05:56 2024

@author: rl05
"""

import sys
import os
import os.path as op
import numpy as np
import glob
import matplotlib.pyplot as plt

from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 
import helper

subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')

color_scheme = {
    'composition': 'Greys',
    'concreteness': 'Blues',
    'concreteness_word': 'Blues',
    'denotation': 'Reds',
    'denotation_cross_condition_test_on_subsective': 'Purples',
    'denotation_cross_condition_test_on_privative': 'Oranges',
    'concreteness_trainWord_testSub': 'Purples',
    'concreteness_trainWord_testPri': 'Oranges',
    'concreteness_general_testSub': 'Purples',
    'concreteness_general_testPri': 'Oranges',
    'specificity': 'Greens',
    'specificity_word': 'Greens'
}


def read_decoding_scores(subjects, analysis, classifier, data_type, roi=None, timegen=False):
    scores_group = []
    for subject in subjects:
        subject = f'sub-{subject}'
        if roi:
            if timegen:
                score_fname = op.join(decoding_dir, f'output/{analysis}/timegen/{classifier}/{data_type}/{subject}/{roi}/scores_time_gen_ROI_{roi}.npy')
            else:
                score_fname = op.join(decoding_dir, f'output/{analysis}/diagonal/{classifier}/{data_type}/{subject}/{roi}/scores_time_decod_ROI_{roi}.npy')
        else:
            if timegen:
                score_fname = op.join(decoding_dir, f'output/{analysis}/timegen/{classifier}/{data_type}/{subject}/scores_time_gen_{data_type}.npy')
            else:
                score_fname = op.join(decoding_dir, f'output/{analysis}/diagonal/{classifier}/{data_type}/{subject}/scores_time_decod_{data_type}.npy')
        if op.exists(score_fname):
            print(f'Decoding scores (analysis: {analysis}) found for {subject}. Loading.')
            scores = np.load(score_fname)
            scores_group.append(scores)
    scores_group = np.stack(scores_group, axis=0)
    print(scores_group.shape)
    return scores_group


def permutation_tests(scores, timegen=False):
    baseline_score = np.full(scores.shape, 0.5) 
    n_subjects = scores.shape[0]
    p_val = 0.05
    df = n_subjects - 1  # degrees of freedom for the test
    t_threshold = t.ppf(1 - p_val / 2, df)
    print(t_threshold)
    scores_diff_group = np.zeros(scores.shape) 
    for s in range(n_subjects):
        scores_diff = np.subtract(scores[s], baseline_score[s])
        scores_diff_group[s,:] = scores_diff
    if timegen is False:
        t_obs, clusters, cluster_pv, H0 = \
            permutation_cluster_1samp_test(scores_diff_group, 
                                           threshold=t_threshold, 
                                           n_permutations=1000, 
                                           seed=42, 
                                           out_type='mask', 
                                           verbose=True)
    else:
        t_obs, clusters, cluster_pv, H0 = \
            spatio_temporal_cluster_1samp_test(scores_diff_group, 
                                               threshold=t_threshold, 
                                               # threshold={'start': 0, 'step':0.1},
                                               n_permutations=1000, 
                                               seed=42, 
                                               out_type='mask', 
                                               verbose=True)
    good_cluster_inds = np.where(cluster_pv < 0.05)[0]
    good_clusters = [(clusters[cluster_ind]) for ii, cluster_ind in enumerate(good_cluster_inds)]
    return good_clusters

def plot_scores(ax, scores, analysis, color):
    if analysis.split('_')[-1] in ['subsective','privative','testSub','testPri']:
        times = np.linspace(0.6, 1.4, scores.shape[1])
        ax.set_xticks(np.arange(0.6, 1.6, 0.2))
    else:
        times = np.linspace(-0.2, 1.4, scores.shape[1])
        ax.set_xticks(np.arange(-0.2, 1.6, 0.2))
        ax.axvline(0., color='lightgrey', alpha=0.7, lw=1, zorder=-20)
        ax.axvline(0.6, color='lightgrey', alpha=0.7, lw=1, zorder=-20)
    avg, sem = helper.calculate_avg_sem(scores)
    ax.plot(times, avg, color=color, lw=2)
    ax.fill_between(times, avg - sem, avg + sem, alpha=.05, color=color)
    ax.axhline(0.5, color='lightgrey', alpha=0.7, linestyle='--', lw=1, zorder=-20)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC') 
    plt.tight_layout()
    return ax

def plot_clusters(clusters, scores, analysis, ax, color):
    avg, _ = helper.calculate_avg_sem(scores)
    if analysis.split('_')[-1] in ['subsective','privative','testSub','testPri']:
        times = np.linspace(0.6, 1.4, scores.shape[1])
        ax.set_xticks(np.arange(0.6, 1.6, 0.2))
    else:
        times = np.linspace(-0.2, 1.4, scores.shape[1])
    for cluster in clusters:
        cluster_start = cluster[0].start
        cluster_stop = cluster[0].stop
        x = times[cluster_start:cluster_stop]
        y1 = avg[cluster_start:cluster_stop]
        y2 = 0.5
        ax.fill_between(x, y1, y2, color=color, alpha=0.3)
        ax.plot(x, y1, color=color, linewidth=3)
    return ax

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"