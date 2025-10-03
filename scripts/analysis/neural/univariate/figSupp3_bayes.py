#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 3 11:00:46 2025

@author: rl05

Plot Supplementary Figure - Bayes Factor time series
"""

import sys
import os
import os.path as op
import pandas as pd
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/preprocessing')
sys.path.append('/imaging/hauk/rl05/fake_diamond/scripts/analysis/neural/decoding')
from config_plotting import *
import fig_constants
import fig_helpers as fh
import config

# 1. CONFIGURATION
figures_dir  = op.join(config.project_repo, f'figures/paper/')

# 3. SET UP FIGURE 
mosaic = [
    ['A','A','B','B'],
    ['.','C','C','.']]
fig, ax_dict = plt.subplot_mosaic(
    mosaic,  # Specify the layout of subplots using the mosaic parameter
    figsize=(fig_constants.FIG_WIDTH, 3.5),  # Set the size of the figure in inches
    dpi=300,  # Set the resolution of the figure in dots per inch
    constrained_layout=True,  # Enable constrained layout for automatic adjustment
    # sharey='row',
    gridspec_kw={
        'height_ratios': [1,1], # Set the relative heights of the rows
        'width_ratios': [1,1,1,1], # Set the relative widths of the columns
        'wspace': 0.001,
        'hspace': 0.005}
)

hemis = ['lh', 'rh', 'lh']
analyses = ['composition','composition','denotation']
rois = ['anteriortemporal-lh', 'anteriortemporal-rh', 'anteriortemporal-lh']
subplot_labels = ['A','B','C']
times = np.linspace(0.0, 0.8, 200)

for analysis, hemi, roi, subplot_label in zip(analyses, hemis, rois, subplot_labels):
    axis = ax_dict[subplot_label]
    bf_dir = f'/imaging/hauk/rl05/fake_diamond/results/neural/bayes/{analysis}'
    bf_ts_fname = os.path.join(bf_dir, f'bayes_factor_time_series_{hemi}.csv')
    if not os.path.isfile(bf_ts_fname):
        continue
    df = pd.read_csv(bf_ts_fname)
    axis.plot(times, df['BF10_A'], color='blue', linewidth=1.25, label='BF10: Concreteness')
    axis.plot(times, df['BF01_A'], color='blue', linestyle='--', linewidth=1.0, label='BF01: Concreteness')
    if analysis == 'composition':
        axis.plot(times, df['BF10_B'], color='black', linewidth=1.25, label='BF10: Composition')
        axis.plot(times, df['BF01_B'], color='black', linestyle='--', linewidth=1.0, label='BF01: Composition')
    elif analysis == 'denotation':
        axis.plot(times, df['BF10_B'], color='black', linewidth=1.25, label='BF10: Denotation')
        axis.plot(times, df['BF01_B'], color='black', linestyle='--', linewidth=1.0, label='BF01: Denotation')

    # Plot interpretability thresholds
    axis.axhline(1, color='gray', linestyle=':', linewidth=0.8)
    axis.axhline(3, color='red', linestyle=':', linewidth=0.8)
    axis.axhline(1/3, color='red', linestyle=':', linewidth=0.8)

    axis.set_xlim(0., 0.8)
    axis.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
    axis.set_xlabel('Time (s)')
    axis.title.set_text(roi)
    if subplot_label in ['A','C']:
        axis.set_ylabel('Bayes Factor\n(log scale)')
    axis.set_yscale('log')

    # plt.grid(True, linestyle=':', alpha=0.5)
    if analysis == 'composition' and hemi == 'lh': # just put legend on the first figure
        axis.legend(loc='upper left', fontsize='x-small')
    if analysis == 'denotation' and hemi == 'lh': # just put legend on the first figure
        axis.legend(loc='lower right', fontsize='x-small')
    
    # add brain model inset
    imgA = mpimg.imread(f'/imaging/hauk/rl05/fake_diamond/figures/labels/fig_roi_label_{roi}_silver.png')
    imagebox = OffsetImage(imgA, zoom=0.03)  # adjust zoom as needed
    ab = AnnotationBbox(
        imagebox,
        xy=(1.0, 1.0),             # upper-right in axis coordinates
        xycoords='axes fraction', # interpret xy as relative to axes
        box_alignment=(1., 1.),     # align image top-right
        frameon=False             # no border around image
    )
    axis.add_artist(ab)

fh.label_panels_mosaic(fig, ax_dict, size = 14)

# plt.suptitle('Main effect of composition in anterior temporal lobe', fontweight='bold')

# 9. FINALIZE & SAVE
out_fname = op.join(figures_dir, 'figSupp3_bayes.png')
plt.savefig(out_fname, dpi=300)
plt.close()
print(f"Saved combined figure to {out_fname}.")
