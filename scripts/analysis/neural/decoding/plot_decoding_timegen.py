#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:55:24 2024

@author: rl05
"""

import sys
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
import config 
import helper
from plot_decoding import *


def demo_locatable_axes_easy(ax, ticks, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.1)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    plt.colorbar(im, cax=ax_cb, ticks=ticks)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)
    
to_plot = input('denotation+concreteness, composition, denotation_cross_condition: ')
if to_plot == 'denotation+concreteness':
    analyses = ['denotation', 'concreteness']
elif to_plot == 'composition':
    analyses = ['composition']
elif to_plot == 'denotation_cross_condition':
    analyses = ['subsective','privative']
classifier = 'logistic'
data_type = 'MEEG'

subjects = config.subject_ids
decoding_dir = op.join(config.project_repo, 'scripts/analysis/neural/decoding')
figures_dir = op.join(config.project_repo, f'figures/decoding/{to_plot}')
if not op.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    
# read in decoding scores
# scores_group = []
# for analysis in analyses:
#     scores = read_decoding_scores(subjects, analysis, classifier, data_type, timegen=True)
#     scores_group.append(scores)
#     del scores
scores_group = []
if to_plot == 'denotation_cross_condition':
    for analysis in analyses:
        scores = read_decoding_scores(subjects, to_plot, classifier, data_type, evaluation=analysis, timegen=True)
        scores_group.append(scores)
        del scores
else:
    for analysis in analyses:
        scores = read_decoding_scores(subjects, analysis, classifier, data_type, timegen=True)
        scores_group.append(scores)
        del scores



if (to_plot == 'denotation+concreteness') or (to_plot == 'denotation_cross_condition'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10., 5.), sharex=True, sharey=True)
    gs = GridSpec(2, 2, width_ratios=[4,6])
    # fig, axd = plt.subplot_mosaic(
    #     """
    #     AB
    #     AC
    #     AD
    #     AE
    #     AF
    #     """,
    #     figsize=(10, 5)
    # ) 
elif to_plot == 'composition':
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5., 5.))

vmaxs = []

for i, analysis in enumerate(analyses):
    
    # if to_plot == 'denotation+concreteness':
    #     fig, axd = plt.subplot_mosaic(
    #         """
    #         AB
    #         AC
    #         AD
    #         AE
    #         AF
    #         """,
    #         figsize=(10, 5)
    #     ) 
        
    if (to_plot == 'denotation+concreteness') or (to_plot == 'denotation_cross_condition'):
        # axis = axes[i, 0]
        axis = axes[i]
        # axis = axd['A']
    elif to_plot == 'composition':    
        axis = axes
    
    # plot temporal generalisation matrix
    group_avg = np.array(scores_group[i]).mean(axis=0)
    vmax = round(np.max(group_avg), 2)
    vmaxs.append(vmax)
    if (to_plot == 'denotation+concreteness') or (to_plot == 'composition'):
        extent = np.array([-0.2,1.4,-0.2,1.4])
    elif to_plot == 'denotation_cross_condition':
        extent = np.array([0.6,1.4,0.6,1.4])
    vmax = np.max(vmaxs)
    im = axis.imshow(group_avg, interpolation=None, origin='lower',
                     cmap=color_scheme[analysis],
                     extent=extent, # times
                     vmin = 0.5, 
                     vmax = vmax
                     )
    
    # some plotting params
    axis.plot(axis.get_xlim(),  axis.get_ylim(), color='lightgrey') # diagonal
    axis.tick_params(axis='both', direction='in')
    # add word onsets
    if (to_plot == 'denotation+concreteness') or (to_plot == 'composition'):
        word_onsets = [0., 0.6]
        for onset in word_onsets:
            axis.axhline(onset, color='lightgrey', linewidth=1, linestyle='--')
            axis.axvline(onset, color='lightgrey', linewidth=1, linestyle='--')
    # add labels
    axis.set_xlabel('Test Time (s)')
    if i == 0:
        axis.set_ylabel('Train Time (s)')

    # add colorbar
    demo_locatable_axes_easy(axis, [0.5,vmax], im)
    
    # # plot cluster extent
    if (to_plot == 'denotation+concreteness') or (to_plot == 'composition'):
        times = np.linspace(-0.2, 1.4, group_avg.shape[0])
    elif to_plot == 'denotation_cross_condition':
        times = np.linspace(0.6, 1.4, group_avg.shape[0])
    X, Y = np.meshgrid(times, times)
    good_clusters = permutation_tests(scores_group[i], timegen=True)
    
    color = plt.get_cmap(color_scheme[analysis])(0.7)
    for cluster in good_clusters:
        # axis.contour(X, Y, cluster, colors=[color])
        axis.contour(X, Y, cluster, colors=['black'])

    
    # # add decomposed testing time courses
    # if to_plot == 'denotation+concreteness':
    #     train_times_idx_to_plot = {'concreteness': [145, 150, 155, 160, 165],
    #                                'denotation': [80, 85, 90, 95, 100]}
    #     for axis_key, train_time_idx in zip(list(axd.keys())[1:], train_times_idx_to_plot[analysis]):
    #         axis = axd[axis_key]
    #         axis.plot(times, group_avg[train_time_idx])
    #         axis.axhline(0.5, color='lightgrey', alpha=0.7, linestyle='--', lw=1, zorder=-20)
    #         axis.set_xlim(-0.2,1.4)
    #         axis.set_ylim(0.48, +0.1)
            
    #     plt.tight_layout()
    #     # plt.savefig(op.join(figures_dir, f'fig_time_gen_group_{classifier}_{data_type}.png'))


# if to_plot == 'composition':
plt.tight_layout()
plt.savefig(op.join(figures_dir, f'fig_time_gen_group_{classifier}_{data_type}.png'))
