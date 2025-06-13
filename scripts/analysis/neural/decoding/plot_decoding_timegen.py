#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:55:24 2024

@author: rl05
"""

import sys
import os
import os.path as op
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config 
import helper
from plot_decoding import *

# Set matplotlib style
mpl.rc_file('fake_diamond.rc')

# ─────────────────────────────────────────────────────────────────────────────
# Parse CLI arguments
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot time-generalisation decoding results")
parser.add_argument('--to_plot', type=str, default='concreteness_xcond', description='Analyses to plot (denotation+concretenss, concretess_xcond)')
parser.add_argument('--data_type', type=str, default='MEEG', description='Sensor or source data (MEEG, ROI)')
args = parser.parse_args()
to_plot = args.to_plot
data_type = args.data_type

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper for colorbars
# ─────────────────────────────────────────────────────────────────────────────
def demo_locatable_axes_easy(ax, ticks, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    cb = plt.colorbar(im, cax=ax_cb, ticks=ticks)
    cb.outline.set_visible(False)
    cb.set_label('AUC', labelpad=-20, rotation=270)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
classifier = 'logistic'
window = 'single' # 'sliding'
micro_ave = True
if micro_ave:
    micro_averaging = 'micro_ave'
roi = input(f'For "{to_plot}", enter ROI: ') if data_type == 'ROI' else None
subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids]

# Figure output directory
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
figures_dir = op.join(config.project_repo, f'figures/decoding/{to_plot}/timegen/{classifier}/{data_type}/{window}/{micro_averaging}')
if not op.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

analysis_map = {
    'denotation+concreteness': ['denotation', 'concreteness'],
    'composition': ['composition'],
    'specificity': ['specificity'],
    'specificity_word': ['specificity_word'],
    'concreteness_xcond': ['concreteness_trainWord_testSub', 'concreteness_trainWord_testPri'],
    'concreteness_xcond_general': ['concreteness_general_testSub', 'concreteness_general_testPri'],
    'concreteness_xcond_full': ['concreteness_trainSub_testSub', 'concreteness_trainSub_testPri',
                                'concreteness_trainPri_testSub', 'concreteness_trainPri_testPri']
}
analyses = analysis_map.get(to_plot, [])
if not analyses:
    raise ValueError(f"Unknown value for --to_plot: {to_plot}")

layout_config = {
    'denotation+concreteness': (2, 1, (3.85, 6)),
    'concreteness_xcond': (2, 1, (3.85, 6)),
    'concreteness_xcond_general': (2, 1, (3.85, 6)),
    'concreteness_xcond_full': (2, 2, (8, 6)),
    'composition': (1, 1, (5, 5)),
    'specificity_word': (1, 1, (5, 5)),
}


# ─────────────────────────────────────────────────────────────────────────────
# Load decoding scores
# ─────────────────────────────────────────────────────────────────────────────
scores_group = []
for analysis in analyses:
    scores = read_decoding_scores(subjects, analysis, classifier, data_type, window=window, roi=roi, timegen=True, micro_ave=micro_ave)
    scores_group.append(scores)
    del scores

# Sampling frequency for plotting
sfreq = int(scores_group[0].shape[1] / 1.6) if to_plot == 'denotation+concreteness' else int(scores_group[0].shape[1] / 0.8)
print('sfreq:', sfreq)

# Time window
is_long_window = to_plot in ['denotation+concreteness', 'composition']
extents = [np.array([0., 1., 0., 1.]) if is_long_window else np.array([0., 0.8, 0., 0.8]),
          np.array([0.6, 1.4, 0.6, 1.4]) if is_long_window else np.array([0., 0.8, 0., 0.8])]
times = [np.linspace(extents[0][0], extents[0][1], scores_group[0][0].shape[0]), 
         np.linspace(extents[1][0], extents[1][1], scores_group[0][0].shape[0])]

# ─────────────────────────────────────────────────────────────────────────────
# Plot each analysis
# ─────────────────────────────────────────────────────────────────────────────
nrows, ncols, figsize = layout_config[to_plot]
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
axes = np.array(axes).reshape(-1)  # flatten for indexing
gs = GridSpec(nrows, ncols)

# get maximum decoding performance score for limiting colorbars
vmaxs = []
for i, analysis in enumerate(analyses):
        
    if to_plot == 'composition':    
        axis = axes
    if to_plot == 'specificity_word':
        axis = axes
    else:
        axis = plt.subplot(gs[i])

    # plot temporal generalisation matrix
    group_avg = np.array(scores_group[i]).mean(axis=0)
    vmax = round(np.max(group_avg), 2)
    vmaxs.append(vmax)
    vmax = np.max(vmaxs)
    im = axis.imshow(group_avg, interpolation=None, origin='lower',
                     cmap=color_scheme[analysis],
                     extent=extents[i], # times
                     vmin = 0.5, vmax = vmax
                     )
    
    # some plotting params
    axis.plot(axis.get_xlim(),  axis.get_ylim(), color='lightgrey', ls='--', lw=1) # diagonal
    axis.tick_params(axis='both', direction='in')
    if to_plot in ['denotation+concreteness', 'concreteness_xcond']:
        axis.set_ylabel('Train Time (s)')
        if i == 1:
            axis.set_xlabel('Test Time (s)') 
    elif (to_plot == 'concreteness_xcond_full'):
        if i in [0,2]:
            axis.set_ylabel('Train Time (s)') 
        if i in [0,1]:
            axis.tick_params(axis='x',labelbottom=False)
        if i in [1,3]:
            axis.tick_params(axis='y',labelleft=False)
        if i in [2,3]:
            axis.set_xlabel('Test Time (s)') 

    # add colorbar
    demo_locatable_axes_easy(axis, [0.5,vmax], im)
    good_clusters, cluster_pv = permutation_tests(scores_group[i], timegen=True, against_chance=True, t_threshold=None)
    print(f'subplot {i+1}')

    X, Y = np.meshgrid(times[i], times[i])
    for cluster, pval in zip(good_clusters, cluster_pv):
        if pval < 0.05:
            axis.contour(X, Y, cluster, colors=['black'], linewidths=1)
        else:
            axis.contour(X, Y, cluster, colors=['grey'], linewidths=1)

plt.tight_layout()
plt.savefig(op.join(figures_dir, f'decode_timegen_{roi}_{sfreq}Hz.png'))
plt.close()