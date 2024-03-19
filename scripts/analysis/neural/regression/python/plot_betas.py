#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:27:58 2024

@author: rl05
"""


import sys
import os
import os.path as op
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from mne import read_source_estimate, read_evokeds, EvokedArray
from mne.stats import permutation_cluster_1samp_test

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config
import helper
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/lm')
from lm_config import *


def plot_betas(ax, betas):
    times = np.linspace(-0.2, 1.4, betas.shape[1])
    avg, sem = helper.calculate_avg_sem(betas)
    ax.plot(times, avg, lw=1)
    ax.fill_between(times, avg - sem, avg + sem, alpha=.1)
    ax.set_ylabel('beta')
    ax.set_xlabel('Time (s)')
    ax.axhline(0., color='lightgrey', alpha=0.7, linestyle='--', lw=1, zorder=-20)
    return ax

def permutation_tests(betas):
    betas = betas[:, 80:]
    baseline = np.full(betas.shape, 0.) 
    n_subjects = betas.shape[0]
    p_val = 0.05
    df = n_subjects - 1  # degrees of freedom for the test
    t_threshold = t.ppf(1 - p_val / 2, df)
    print(t_threshold)
    betas_diff_group = np.zeros(betas.shape) 
    for s in range(n_subjects):
        betas_diff = np.subtract(betas[s], baseline[s])
        betas_diff_group[s,:] = betas_diff
    t_obs, clusters, cluster_pv, H0 = \
        permutation_cluster_1samp_test(betas_diff_group, 
                                       threshold=t_threshold, 
                                       n_permutations=1000, 
                                       seed=42, 
                                       out_type='mask', 
                                       verbose=True)
    good_cluster_inds = np.where(cluster_pv < 0.05)[0]
    good_clusters = [(clusters[cluster_ind]) for ii, cluster_ind in enumerate(good_cluster_inds)]
    return good_clusters

def plot_clusters(ax, clusters, betas):
    betas = betas[:, 80:]
    avg, _ = helper.calculate_avg_sem(betas)
    times = np.linspace(-0.2, 1.4, betas.shape[1])
    for cluster in clusters:
        cluster_start = cluster[0].start
        cluster_stop = cluster[0].stop
        x = times[cluster_start:cluster_stop]
        y1 = avg[cluster_start:cluster_stop]
        y2 = 0.
        ax.fill_between(x, y1, y2, alpha=0.3)
        ax.plot(x, y1, linewidth=3)
    return ax



subject_ids = config.subject_ids
print(f'subjects (n={len(subject_ids)}): \n', subject_ids)

data_dir = op.join(config.project_repo, 'data')
results_dir = op.join(config.project_repo, 'results')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir


data_type = input('sensor or source: ')

analysis_id = input('analysis id: ')
analysis = analysis_mapping[analysis_id][0]
print('analysis: ', analysis)

output_dir = op.join(config.project_repo, 'scripts/analysis/neural/lm/betas')

predictors_mapping = {
    '1': ['denotation','concreteness','composition','composition:concreteness'],
    '2': ['composition','specificity'],
    }
predictors = predictors_mapping[analysis_id]
ncol = len(predictors)
rois = ['anteriortemporal-lh','posteriortemporal-lh',
       'inferiorfrontal-lh','temporoparietal-lh','lateraloccipital-lh']
nrow = len(rois)

for predictor in predictors:
    
    if data_type == 'source':
        fig, axes = plt.subplots(nrow, 1, figsize=(7,2*nrow), sharex=True, tight_layout=True)

        for i, (roi, axis)  in enumerate(zip(rois, axes)):
            
            betas_group = []
            for subject_id in subject_ids:
            
                subject = f'sub-{subject_id}'
                # print(subject)
                
                subject_output_dir = op.join(output_dir, analysis, data_type, subject)
                if not op.exists(subject_output_dir):
                    pass
                else:
                    betas_fname = op.join(subject_output_dir, f'betas_{roi}_{predictor}-lh.stc')
                    if not op.exists(betas_fname):
                        pass
                    else:
                        betas = read_source_estimate(betas_fname)
                        betas_group.append(betas.data.mean(axis=0))
            betas_group = np.stack(betas_group)
            print(betas_group.shape)
            # ax = axis[i]
            plot_betas(axis, betas_group)
            
            good_clusters = permutation_tests(betas_group)
            plot_clusters(axis, good_clusters, betas_group)

            axis.set_title(roi)
            
        fig.suptitle(predictor, fontsize=14)

        plt.savefig(op.join(results_dir, f'neural/roi/lm/betas_source_{analysis}_{predictor}.png'))
    
    elif data_type == 'sensor':
        # fig, axis = plt.subplots(1, 1, figsize=(7,4), sharex=True, tight_layout=True)

        betas_group = []
        for subject_id in subject_ids:
            subject = f'sub-{subject_id}'
            print(subject)
            
            subject_output_dir = op.join(output_dir, analysis, data_type, subject)
            if not op.exists(subject_output_dir):
                pass
            else:
                betas_fname = op.join(subject_output_dir, f'betas_sensor_{predictor}-ave.fif')
                if not op.exists(betas_fname):
                    pass
                else:
                    betas = read_evokeds(betas_fname, verbose=False)[0]
                    betas_group.append(betas.data)
                    print(betas.data.shape)
        betas_group = np.stack(betas_group)
        print(betas_group.shape)

        info = betas.info
        info['bads'] = []
        betas_group_epo = EvokedArray(betas_group.mean(axis=0), info, tmin=-0.5)
        betas_group_epo.plot_joint(times=[0.17, 0.4, 0.6, 1.0, 1.2])


        plt.savefig(op.join(results_dir, f'neural/roi/lm/betas_sensor_{analysis}_{predictor}.png'))
