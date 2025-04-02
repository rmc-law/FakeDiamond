#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:26:56 2024

@author: rl05

Plot uncorrected t-maps for each contrast 

TO PLOT THIS, NEED TO RUN `module load mnelib` ON CLUSTER FIRST FOR PLOTTING
"""


import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator, make_axes_locatable

from mne import (read_epochs, read_source_spaces, read_labels_from_annot,
                 read_source_estimate)
from mne.viz import plot_brain_colorbar
from mne.stats import ttest_1samp_no_p

import config 
from helper import calculate_avg_sem

subjects = [f'sub-{subject_id}' for subject_id in config.subject_ids if subject_id not in ['16']]
print(f'subjects (n={len(subjects)}): ', subjects)
data_dir = op.join(config.project_repo, 'data')
figs_dir = op.join(config.project_repo, 'figures')
preprocessed_data_path = op.join(data_dir, 'preprocessed')
subjects_dir = op.join(data_dir, 'mri')
os.environ['SUBJECTS_DIR'] = subjects_dir
fsaverage_src_fname = op.join(subjects_dir, 'fsaverage_src', 'fsaverage_src_oct6_src.fif')
src_fsaverage = read_source_spaces(fsaverage_src_fname, verbose=False)
stc_path = op.join(data_dir, 'stcs')
analysis = input('analysis (composition, denotation): ')
results_dir = '/imaging/hauk/rl05/fake_diamond/figures/univariate/tmaps/'


#%% set figure style 
FONT = 'Arial'
FONT_SIZE = 15
LINEWIDTH = 1.5
EDGE_COLOR = 'grey'
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    'axes.labelsize': FONT_SIZE,
    'axes.edgecolor': EDGE_COLOR,
    'axes.linewidth': LINEWIDTH,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1,
    'ytick.major.width': 1
})


#%% plot tmaps

if analysis == 'composition':
    
    conditions = ['concrete-baseline','abstract-baseline','concrete-subsective','abstract-subsective']

    stcs_conditions = []
    for condition in conditions:
        print('Reading condition: ', condition)
        stcs_condition = np.zeros((len(subjects),8196,475))
        for i_subject, subject in enumerate(subjects):
            stcs_path = op.join(data_dir, 'stcs', subject)
            stcs_fname = op.join(stcs_path, f'{subject}_{condition}_MEEG-lh.stc')
            stcs = read_source_estimate(stcs_fname)
            stcs_condition[i_subject,:,:] = stcs.data
        stcs_conditions.append(stcs_condition)

    # calculate condition difference
    stcs_word = np.mean(stcs_conditions[:2], axis=0)
    stcs_phrase = np.mean(stcs_conditions[2:], axis=0)
    X = stcs_phrase - stcs_word

elif analysis == 'denotation':
    
    conditions = ['concrete-subsective','abstract-subsective','concrete-privative','abstract-privative']

    stcs_conditions = []
    for condition in conditions:
        print('Reading condition: ', condition)
        stcs_condition = np.zeros((len(subjects),8196,475))
        for i_subject, subject in enumerate(subjects):
            stcs_path = op.join(data_dir, 'stcs', subject)
            stcs_fname = op.join(stcs_path, f'{subject}_{condition}_MEEG-lh.stc')
            stcs = read_source_estimate(stcs_fname)
            stcs_condition[i_subject,:,:] = stcs.data
        stcs_conditions.append(stcs_condition)

    # calculate condition difference
    stcs_subsective = np.mean(stcs_conditions[:2], axis=0)
    stcs_privative = np.mean(stcs_conditions[2:], axis=0)
    X = stcs_subsective - stcs_privative
    
ts = ttest_1samp_no_p(X)

dummy_stc = stcs.copy()
dummy_stc._data = ts

dummy_stc = dummy_stc.crop(tmin=0.6)
dummy_stc = dummy_stc.bin(width=0.0995, tstop=1.396)
for hemi in ['lh','rh']:
    screenshots = []
    for initial_time in np.arange(0.65,1.45,0.1):
        brain = dummy_stc.plot(subject='fsaverage_src',
                                surface='partially_inflated',
                                size=800,
                                smoothing_steps=10,
                                hemi=hemi,
                                views='lat',
                                initial_time=initial_time,
                                cortex='low_contrast', background='white',
                                show_traces=False, colorbar=False, time_viewer=False)
        screenshot = brain.screenshot()
        brain.close()

        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        screenshots.append(cropped_screenshot)

    # figsize unit is inches
    fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(16, 2), dpi=300)

    # add tmaps for each time tick
    time_windows = ['0-0.1 s','0.1-0.2 s','0.2-0.3 s','0.3-0.4 s','0.4-0.5 s','0.5-0.6 s','0.6-0.7 s','0.7-0.8 s']
    for tmap_idx in range(len(screenshots)):
        axes[tmap_idx].imshow(screenshots[tmap_idx])
        axes[tmap_idx].axis("off")
        axes[tmap_idx].title.set_text(time_windows[tmap_idx])

    # add a vertical colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes[tmap_idx+1])
    axes[tmap_idx+1].axis("off")
    cax = divider.append_axes("left", size="5%", pad=0.2)
    if analysis == 'composition':
        pos_lims = [1.89006821,2.06368817,3.3993154] # get this from plotting stc and see what mne uses at control points 
    if analysis == 'denotation':
        pos_lims = [1.8973722,2.05492463,3.05123078]
    cbar = plot_brain_colorbar(cax, clim=dict(kind='value',pos_lims=pos_lims), colormap='auto', label="$\it{t}$-values")
    pos_lims = [round(value, 1) for value in pos_lims]
    cbar.set_ticks(ticks=[-pos_lims[0], -(pos_lims[0]+pos_lims[2])/2, -pos_lims[2], 0,
                          pos_lims[0], (pos_lims[0]+pos_lims[2])/2, pos_lims[2]])


    # tweak and save fig
    plt.tight_layout()
    plt.savefig(op.join(figs_dir, f'univariate/tmaps/{analysis}_{hemi}_tmaps.png'))
    plt.close()