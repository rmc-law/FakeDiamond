#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:58:06 2024

@author: rl05
"""

import sys
import os
import os.path as op
import numpy as np
from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test, summarize_clusters_stc
import mne

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config 

subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
data_dir = op.join(config.project_repo, 'data')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

fsaverage_src_fname = op.join(data_dir, 'mri', 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src = mne.read_source_spaces(fsaverage_src_fname) 

analysis = 'composition'

# figures_dir = op.join(decoding_dir, f'figures/{analysis}')
# if not op.exists(figures_dir):
#         os.makedirs(figures_dir, exist_ok=True)


#%% plot group-average of source localised 'decoding evoked responses'

coef_stc_group = []

for subject in subjects:
    
    subject = f'sub-{subject}'
    coef_dir = op.join(decoding_dir, f'output/{analysis}/diagonal/logistic/MEEG/{subject}')
    coef_fname = op.join(coef_dir, 'scores_coef_MEEG.npy')
    
    coef_projection_fname = op.join(coef_dir, f'source_projection_coef_{analysis}-lh.stc')
    if op.exists(coef_projection_fname):
        stc = mne.read_source_estimate(coef_projection_fname)
        coef_stc_group.append(stc._data)
        
coef_stc_group = np.array(coef_stc_group).mean(axis=0)
dummy_stc = stc.copy() # for convenience: getting info
dummy_stc._data = coef_stc_group # replace with group level average coef projection
# dummy_stc.tmin = -0.5
dummy_stc.tmin = 0.6
dummy_stc.tstep = 0.01
dummy_stc.plot(subject='fsaverage_src', hemi='both')
# dummy_stc.save_movie()


#%% perform cluster-based permutation test 


if analysis.startswith('denotation_cross'):
    tmin = 0.6,
else:
    tmin = -0.5
tstep = 0.01

coef_stc_group = []

for subject in subjects:
    
    subject = f'sub-{subject}'
    coef_dir = op.join(decoding_dir, f'output/{analysis}/diagonal/logistic/MEEG/{subject}')
    coef_fname = op.join(coef_dir, 'scores_coef_MEEG.npy')
    
    coef_projection_fname = op.join(coef_dir, f'source_projection_coef_{analysis}-lh.stc')
    if op.exists(coef_projection_fname):
        stc = mne.read_source_estimate(coef_projection_fname)
        coef_stc_group.append(stc._data)
        
X = np.array(coef_stc_group)

n_subjects, n_vertices_sample, n_times = X.shape

baseline = np.zeros(X.shape) 
p_val = 0.05
df = n_subjects - 1  # degrees of freedom for the test
t_threshold = t.ppf(1 - p_val / 2, df)
print(t_threshold)

print("Computing adjacency.")
adjacency = mne.spatial_src_adjacency(src)
fsave_vertices = [s["vertno"] for s in src]

# Note that X needs to be a multi-dimensional array of shape
# observations (subjects) × time × space, so we permute dimensions
X = np.transpose(X, [0, 2, 1])

t_obs, clusters, cluster_pv, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, 
                                       threshold=t_threshold, 
                                       n_permutations=1000,
                                       # tail=1, 
                                       adjacency=adjacency,
                                       n_jobs=-1,
                                       seed=42, 
                                       # max_step=1,
                                       # out_type='mask', 
                                       verbose=True)
good_cluster_inds = np.where(cluster_pv < 0.05)[0]
good_clusters = [(clusters[cluster_ind]) for ii, cluster_ind in enumerate(good_cluster_inds)]


stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, vertices=fsave_vertices, subject="fsaverage_src")
brain = stc_all_cluster_vis.plot(
    hemi="both",
    views="lateral",
    time_label="temporal extent (ms)",
    size=(800, 800),
    smoothing_steps=5
    # clim=dict(kind="value", pos_lims=[0, 1, 40]),
)
