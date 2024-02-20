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

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 
import helper

subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')

analysis = input('analysis (composition, concreteness(_word), denotation, specificity(_word), denotation_cross_condition): ')
classifier = input('classifier (logistic, svc, naive_bayes): ')
data_type = input('input_features (MEEG, MEG, source): ')
if analysis == 'denotation_cross_condition':
    # evaluation = input('test on (subsective, privative): ')
    evaluations = ['subsective','privative']

figures_dir = op.join(decoding_dir, f'figures/{analysis}')
if not op.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)


def read_decoding_scores(subjects, analysis, classifier, data_type, evaluation=None, *args, **kwargs):
    scores_group = []
    for subject in subjects:
        subject = f'sub-{subject}'
        if analysis == 'denotation_cross_condition':
            ending_array = f'_test-on-{evaluation}.npy'
        else:
            ending_array = '.npy'
        decoding_score_array = op.join(decoding_dir, f'output/{analysis}/{classifier}/{data_type}/{subject}/scores_time_decod_{data_type}'+ending_array)
        if op.exists(decoding_score_array):
            print(f'Decoding scores (analysis: {analysis}) found for {subject}. Loading.')
            scores = np.load(decoding_score_array)
            scores_group.append(scores)
    scores_group = np.stack(scores_group, axis=0)
    print(scores_group.shape)
    return scores_group


def plot_scores(fig, ax, times, avg, sem, analysis, evaluation=None, *args, **kwargs):
    if evaluation is not None:
        label = evaluation
    else:
        label = 'score'
    ax.plot(times, avg, label=label)
    ax.fill_between(times, avg - sem, avg + sem, alpha=.1)
    ax.axhline(0.5, color='lightgrey', alpha=0.7, linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('Decoding score (AUC)') 
    ax.legend()
    if analysis == 'lexicality':
        ax.set_xticks(np.arange(-0.2, 0.6, 0.2))
    elif analysis == 'denotation_cross_condition':
        ax.set_xticks(np.arange(0.6, 1.4, 0.2))
    else:
        ax.set_xticks(np.arange(-0.2, 1.4, 0.2))
    plt.tight_layout()
    return fig, ax


# read in decoding scores
if analysis == 'denotation_cross_condition':
    scores_group_subsective = read_decoding_scores(subjects, analysis, classifier, data_type, evaluation='subsective')
    scores_group_privative = read_decoding_scores(subjects, analysis, classifier, data_type, evaluation='privative')
    scores_group = [scores_group_subsective, scores_group_privative]
else:
    scores_group = read_decoding_scores(subjects, analysis, classifier, data_type)


# plot group decoding scores
if analysis == 'denotation_cross_condition':
    stats = [helper.calculate_avg_sem(s) for s in scores_group]
else:
    stat = helper.calculate_avg_sem(scores_group)
fig, ax = plt.subplots()
if analysis == 'denotation_cross_condition':
    times = np.linspace(0.6, 1.4, stats[0][0].shape[0])
    plot_scores(fig, ax, times, *stats[0], analysis=analysis, evaluation='subsective')
    plot_scores(fig, ax, times, *stats[1], analysis=analysis, evaluation='privative')
    plt.title(f'Temporal decoding of {analysis} (subject n={len(scores_group[0])})')
else:
    times = np.linspace(-0.2, 1.4, stat[0].shape[0])
    plot_scores(fig, ax, times, *stat, analysis=analysis)
    plt.title(f'Temporal decoding of {analysis} (subject n={len(scores_group)})')
plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{data_type}.png'))
