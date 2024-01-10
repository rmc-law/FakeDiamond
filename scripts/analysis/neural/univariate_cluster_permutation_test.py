#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:13:01 2023

@author: rl05
"""

import os
import os.path as op
import sys
import pickle

import numpy as np
# import pandas as pd

from mne import read_source_spaces, read_source_estimate, spatial_src_adjacency
from mne.stats import (
    f_oneway,
    f_mway_rm,
    f_threshold_mway_rm,
    spatio_temporal_cluster_test,
    summarize_clusters_stc,
)
# from eelbrain import Dataset, load, Factor, plot, testnd

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
import config

subjects = config.subject_ids
data_dir = op.join(config.project_repo, 'data')
results_dir = op.join(config.project_repo, 'results')

preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir

subjects_to_ignore = [
    '17',
    '24'
]

subjects = [subject for subject in subjects if subject not in subjects_to_ignore]
print(f'subjects (n={len(subjects)}): ', subjects)

ch_type = input('MEEG or MEG: ')
experiment = input('compose or specificity: ')
parc = 'fake_diamond'

conditions = [
    'concrete-subsective',
    'concrete-privative',
    'concrete-baseline',
    'abstract-subsective',
    'abstract-privative',
    'abstract-baseline',
    'low',
    'mid',
    'high'
]


src_fname = op.join(data_dir, 'mri', 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src = read_source_spaces(src_fname) 
fsave_vertices = [src[0]["vertno"], src[1]["vertno"]]


stcs = dict()

for condition in conditions:
    print(f'Reading in source estimates from condition {condition}.')
    stcs[condition]= []
    for subject in subjects:
        subject = f'sub-{subject}' 
        stc_fname = op.join(data_dir, 'stcs', subject, f'{subject}_{condition}_{ch_type}-lh.stc')        
        stc = read_source_estimate(stc_fname)
        stcs[condition].append(stc)


# prepare the group matrix for permutation tests 
data_experiment = []
# get the relevant conditions for the relevant experiment
if experiment == 'compose':
    relevant_conditions = conditions[:6]
elif experiment == 'specificity':
    relevant_conditions = conditions[6:]
n_conditions = len(relevant_conditions)
# stack arrays by condition
for relevant_condition in relevant_conditions:
    x = np.stack([stcs[relevant_condition][i].data for i in range(len(stcs[relevant_condition]))])
    data_experiment.append(x)
X = np.stack(data_experiment, axis =-1) # shape (n_sub, n_source, n_time, n_cond)
# X = np.abs(X)
# X needs to be a list of multi-dimensional arrays (one per condition) of shape: samples (subjects) × time × space.
X = np.transpose(X, [0, 2, 1, 3]) # shape (n_sub, n_time, n_source, n_cond)
# split the array into a list of conditions and discard the empty dimension resulting from the split using numpy squeeze.
X = [np.squeeze(x) for x in np.split(X, n_conditions, axis=-1)] 

print('Computing adjacency.')
adjacency = spatial_src_adjacency(src)


# Here we set a cluster forming threshold based on a p-value for
# the cluster based permutation test.
# We use a two-tailed threshold, the "1 - p_threshold" is needed
# because for two-tailed tests we must specify a positive threshold.
p_threshold = 0.05
df = len(subjects) - 1  # degrees of freedom for the test

# Now let's actually do the clustering. This can take a long time...
print('Clustering.')

if experiment == 'compose':
    factor_levels = [2, 3]
    effects = 'A:B'
    def stat_fun(*args):
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                         effects=effects, return_pvals=False)[0]
elif experiment == 'specificity':
    factor_levels = [3]
    effects = 'A'
    def stat_fun(*args):
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                         effects=effects, return_pvals=False)[0]
# Tell the ANOVA not to compute p-values which we don't need for clustering
return_pvals = False

# a few more convenient bindings
n_times = X[0].shape[1]
n_subjects = len(subjects)
n_permutations = 1000



pthresh = 0.005
f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects, pthresh)

F_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_test(X, adjacency=adjacency, n_jobs=-1,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations,
                                 buffer_size=None)
    
with open(op.join(results_dir, f'univariate_rm_anova_permutation_cluster_{experiment}_{effects}.P'), 'wb') as f:
    pickle.dump(clu, f)


# Now select the clusters that are sig. at p < 0.05 (note that this value
# is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


print("Visualizing clusters.")

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
tstep = stc.tstep * 1000
stc_all_cluster_vis = summarize_clusters_stc(
    clu, tstep=tstep, vertices=fsave_vertices, subject="fsaverage"
)

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration

# The brighter the color, the stronger the interaction between
# stimulus modality and stimulus location

brain = stc_all_cluster_vis.plot(
    subjects_dir=subjects_dir,
    views="lat",
    time_label="temporal extent (ms)",
    clim=dict(kind="value", lims=[0, 1, 40]),
)
brain.save_image(f'univariate_rm_anova_permutation_cluster_{experiment}.png')
brain.show_view("medial")


inds_t, inds_v = [
    (clusters[cluster_ind]) for ii, cluster_ind in enumerate(good_cluster_inds)
][
    0
]  # first cluster

times = np.arange(X[0].shape[1]) * tstep * 1e3

import matplotlib.pyplot as plt

plt.figure()
colors = plt.cm.Greens([0.3,0.5,0.75])
event_ids = ['low','mid','high']

for ii, (condition, color, eve_id) in enumerate(zip(X, colors, event_ids)):
    # extract time course at cluster vertices
    condition = condition[:, :, inds_v]
    # normally we would normalize values across subjects but
    # here we use data from the same subject so we're good to just
    # create average time series across subjects and vertices.
    mean_tc = condition.mean(axis=2).mean(axis=0)
    std_tc = condition.std(axis=2).std(axis=0)
    plt.plot(times, mean_tc.T, color=color, label=eve_id)
    plt.fill_between(
        times, mean_tc + std_tc, mean_tc - std_tc, color=color, alpha=0.2, label=""
    )

ymin, ymax = mean_tc.min() - 0, mean_tc.max() + 0
plt.xlabel("Time (ms)")
plt.ylabel("Activation (F-values)")
plt.xlim(times[[0, -1]])
plt.ylim(ymin, ymax)
plt.fill_betweenx(
    (ymin, ymax), times[inds_t[0]], times[inds_t[-1]], color="orange", alpha=0.3
)
plt.legend()
plt.title("Interaction between stimulus-modality and location.")
plt.show()


# # create eelbrain dataset

# stcs = []
# subject_list = []
# condition_list = []

# for subject in subjects:
#     subject = f'sub-{subject}'
#     print(f'Reading in source estimates from {subject}.')
#     stc_path = op.join(data_dir, 'stcs', subject)
#     for condition in conditions:
#         condition = condition.replace('/','-')
#         stc = read_source_estimate(op.join(stc_path, f'{subject}_{condition}_{ch_type}-lh.stc'),
#                                    subject='fsaverage')
#         stcs.append(stc)
#         subject_list.append(subject)
#         condition_list.append(condition)

# ds = Dataset()

# if experiment == 'compose':
#     condition = [condition.split('-')[0]+'-'+condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
#     concreteness = [condition.split('-')[0] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
#     denotation = [condition.split('-')[1] for condition in condition_list if condition.startswith(('concrete', 'abstract'))]
# elif experiment == 'specificity':
#     specificity = [condition.split('-')[0] for condition in condition_list if condition.startswith(('low','mid','high'))]

# ds['stcs'] = load.fiff.stc_ndvar(stcs, 
#                                  subject='fsaverage_src', 
#                                  src='oct-6', 
#                                  parc=parc) 
# ds['subject'] = Factor(subject_list,random=True)

# if experiment == 'compose':    
#     ds['condition'] = Factor(condition)
#     ds['condition'].sort_cells(['concrete-baseline',
#                                 'concrete-subsective',
#                                 'concrete-privative',
#                                 'abstract-baseline',
#                                 'abstract-subsective',
#                                 'abstract-privative'         
#                                 ])
#     ds['concreteness'] = Factor(concreteness)
#     ds['concreteness'].sort_cells(['concrete','abstract'])
#     ds['denotation'] = Factor(denotation)
#     ds['denotation'].sort_cells(['baseline','subsective','privative'])
# elif experiment == 'specificity':
#     ds['specificity'] = Factor(specificity)
#     ds['specificity'].sort_cells(['low','mid','high'])
# stc_reset = ds['stcs']


# '''.............................. run ROI test ..............................'''

# rois = ['anteriortemporal-lh', 'posteriortemporal-lh',
#         'inferiorfrontal-lh', 'temporoparietal-lh']

# for roi in rois:
#     ds['stcs'] = stc_reset
#     stc_region = stc_reset.sub(source=roi) # subset language network region data
#     ds['stcs'] = stc_region # assign this back to the ds

#     # perform temporal permutation test in a particular region
#     res = testnd.anova(y=ds['stcs'].mean('source'), 
#                        x='concreteness*denotation*Subject', 
#                        data=ds, 
#                        samples=5000, 
#                        pmin=0.05,
#                        tfce=True,
#                        match='Subject')

#     pickle.dump(res, open(op.join(results_dir, f'{roi}.pickle'), 'wb'))

#     f = open(op.join(results_dir, f'{roi}_results_table.txt'), 'w')
#     f.write('Model: %s, N=%s\n' %(res.x, len(subjects)))
#     f.write('tstart=%s, tstop=%s, samples=%s, pmin=%s, mintime=??\n\n' %(res.tstart, res.tstop, res.samples, res.pmin))
#     f.write(str(res.clusters))
#     f.close()

#     pmin = 0.05
#     mask_sign_clusters = np.where(res.clusters['p'] <= pmin, True, False)
#     sign_clusters = res.clusters[mask_sign_clusters]

#     if sign_clusters.n_cases != None: #check for significant clusters
#         for i in range(sign_clusters.n_cases):
#             cluster_nb = i+1
#             cluster = sign_clusters[i]['cluster']
#             tstart = sign_clusters[i]['tstart']
#             tstop = sign_clusters[i]['tstop']
#             effect = sign_clusters[i]['effect']

#             effect = effect.replace(' x ', '%') # Changing format of string for interaction effects.

#             print('Plotting time series for %s' %region)
#             timecourse = src_region.mean('source')
#             activation = eelbrain.plot.UTSStat(timecourse, effect, ds=ds, error='sem', match='Subject', legend='lower left', xlabel='Time (ms)', ylabel='Activation (dSPM)', xlim=(0,0.6), title='Cluster %s: Effect of %s at %s' %(i+1, effect, region))
#             activation.add_vspan(xmin=tstart, xmax=tstop, color='lightgrey', zorder=-50, alpha=0.4)
#             # activation.add_vline(0, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
#             # activation.add_vline(3, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
#             # activation.add_vline(3.6, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
#             # activation.add_vline(4.2, color='red', alpha=0.8, linestyle=':', linewidth=0.7)
#             activation._axes[0].set_xticks([0,tstart,tstop])

#             activation.save(os.path.join(output, 'clus%s_%s_%s_(%s-%s).png' %(i+1, tstart, tstop, effect, region)), dpi=250)
#             activation.close()

#             ds['average_source_activation'] = timecourse.mean(time=(tstart,tstop)) #!!! should be restrained to my sign timewindow
#             bar = plot.Barplot(ds['average_source_activation'], effect, ds=ds, title='Average activation at %s' %region, match='Subject', ylabel='Average source activation (dSPM)')
#             bar.save(os.path.join(output, 'cluster%s_BarGraph_(%s-%s)_effect=%s.png'%(i+1, tstart, tstop, effect)))
#             bar.close()
