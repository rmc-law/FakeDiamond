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
input_features = input('input_features (MEEG, MEG, source): ')
if analysis == 'denotation_cross_condition':
    evaluation = input('test on (subsective, privative): ')

figures_dir = op.join(decoding_dir, f'figures/{analysis}')
if not op.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)


# read in decoding scores
scores_all = []

for subject in subjects:
    
    subject = f'sub-{subject}'
    if analysis == 'denotation_cross_condition':
        ending_array = f'_test-on-{evaluation}.npy'
    else:
        ending_array = ''
    decoding_score_array = op.join(decoding_dir, f'output/{analysis}/{classifier}/{input_features}/{subject}/scores_time_decod_{input_features}'+ending_array)

    if op.exists(decoding_score_array):
        print(f'Decoding scores (analysis: {analysis}) found for {subject}. Loading.')
        scores = np.load(decoding_score_array)
        scores_all.append(scores)

scores_all = np.stack(scores_all, axis=0)
print(scores_all.shape)


# plot group decoding scores
avg, sem = helper.calculate_avg_sem(scores_all)
if analysis == 'lexicality':
    times = np.linspace(-0.2, 0.6, avg.shape[0])
elif analysis == 'denotation_cross_condition':
    times = np.linspace(0.6, 1.4, avg.shape[0])
else:
    times = np.linspace(-0.2, 1.4, avg.shape[0])

fig, axis = plt.subplots()
axis.plot(times, avg, label='accuracy')
axis.fill_between(times, avg - sem, avg + sem, alpha=.1)
axis.axhline(0.5, color='lightgrey', alpha=0.7, linestyle='--', label='chance')
if analysis == 'lexicality':
    axis.set_xticks(np.arange(-0.2, 0.6, 0.2))
elif analysis == 'denotation_cross_condition':
    axis.set_xticks(np.arange(0.6, 1.4, 0.2))
else:
    axis.set_xticks(np.arange(-0.2, 1.4, 0.2))
axis.set_ylabel('Decoding score (AUC)') 
axis.set_xlabel('Time (s) relative to word onset')
plt.title(f'Temporal decoding of {analysis} (subject n={scores_all.shape[0]})')
plt.legend()
if analysis == 'denotation_cross_condition':
    ending_fig = f'_test-on-{evaluation}.png'
else:
    ending_fig = '.png'
plt.savefig(op.join(figures_dir, f'fig_time_decod_group_{classifier}_{input_features}'+ending_fig))
